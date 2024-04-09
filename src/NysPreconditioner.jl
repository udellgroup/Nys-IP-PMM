using LinearAlgebra
using FameSVD
using LinearOperators

"""
    mutable struct Sketch_ATΩ{T}

Structure to store the cache and the sketch AᵀΩ where Ω = 1/√m * randn(T, m, r).
Later used for computing the Nystrom sketch of normal equation ADAᵀ.
"""
Base.@kwdef mutable struct Sketch_ATΩ{T}
    Ω::AbstractMatrix{T}        # Ω = 1/√m * randn(T, m, r)
    AᵀΩ:: AbstractMatrix{T}
    DAᵀΩ:: AbstractMatrix{T}    # just for cache
    Y::AbstractMatrix{T}        # just for cache: Y = ADAᵀΩ
    Z::AbstractMatrix{T}        # just for cache: Z = Ω'Y
end

"""
Sketch_ATΩ Constructor

    Sketch_ATΩ(opAT::Union{LinearOperator{T}, AdjointLinearOperator{T}}, r::Int)

Given a linear operator opAT and sketch size r, construct Sketch_ATΩ with Gaussian test matrix Ω = 1/√m * randn(T, m, r).
"""
function Sketch_ATΩ(opAT::Union{LinearOperator{T}, AdjointLinearOperator{T}}, r::Int) where T
    ncols, nrows = size(opAT)
    Ω = inv(sqrt(nrows)) * randn(T, nrows, r)
    AᵀΩ = zeros(T, ncols, r)
    for i in 1:r
        @views mul!(AᵀΩ[:, i], opAT, Ω[:, i])
    end
    return Sketch_ATΩ(Ω, AᵀΩ, similar(AᵀΩ), zeros(T, nrows, r), zeros(T, r, r))
end

"""
Sketch_ATΩ Constructor

    Sketch_ATΩ(opN::opNormalEquations, r::Int)

Given a opNormalEquations opN = ADAᵀ and sketch size r, construct Sketch_ATΩ with opAᵀ and r.
"""
Sketch_ATΩ(opN::opNormalEquations, r::Int) = Sketch_ATΩ(opN.opAT, r)


"""
Sketch_ATΩ Constructor

    Sketch_ATΩ{T}(initializer::UndefInitializer)

Initialize Sketch_ATΩ with undefined Ω, AᵀΩ, DAᵀΩ, Y, Z. 
"""
function Sketch_ATΩ{T}(initializer::UndefInitializer) where T 
    return Sketch_ATΩ(Array{T}(initializer, (0,0)), 
                      Array{T}(initializer, (0,0)), 
                      Array{T}(initializer, (0,0)), 
                      Array{T}(initializer, (0,0)), 
                      Array{T}(initializer, (0,0)))
end


# Define basic properties
Base.isassigned(sketch_ATΩ::Sketch_ATΩ) = isassigned(sketch_ATΩ.Ω)


##
"""
    mutable struct NystromSketch_mutable{T}

Mutable version of NystromSketch
"""
mutable struct NystromSketch_mutable{T}
    U::AbstractMatrix{T}
    Λ::Diagonal{T, Vector{T}}
    
    function NystromSketch_mutable(U::AbstractMatrix{T}, Λ::Diagonal{T, Vector{T}}) where T
        @assert size(U, 2) == size(Λ, 1) "size(U, 2) and size(Λ, 1) must have the same size."
        return new{T}(U, Λ)
    end
end

function NystromSketch_mutable{T}(initializer::UndefInitializer) where T 
    return NystromSketch_mutable(Array{T}(initializer, (0,0)), 
                                 Diagonal(Vector{T}(initializer, 0)))
end


# Define basic properties
Base.eltype(::NystromSketch_mutable{T}) where {T} = T
Base.size(Ahat::NystromSketch_mutable) = (size(Ahat.U, 1), size(Ahat.U, 1))
Base.size(Ahat::NystromSketch_mutable, d::Int) = d <= 2 ? size(Ahat)[d] : 1
Base.isassigned(Ahat::NystromSketch_mutable) = isassigned(Ahat.U)
LinearAlgebra.rank(Ahat::NystromSketch_mutable) = size(Ahat.U, 2)
LinearAlgebra.svdvals(Ahat::NystromSketch_mutable) = Ahat.Λ.diag


"""
    update_NystromSketch!(Nys_sketch, sketch_ATΩ, opN)

Update the Nystrom sketch of normal equation opN = ADAᵀ after D changes, using AᵀΩ cached in sketch_ATΩ.
"""
function update_NystromSketch!(Nys_sketch::NystromSketch_mutable{T}, sketch_ATΩ::Sketch_ATΩ{T}, opN::opNormalEquations{T}) where T
    Ω = @views sketch_ATΩ.Ω
    DAᵀΩ = @views sketch_ATΩ.DAᵀΩ
    Y = @views sketch_ATΩ.Y
    Z = @views sketch_ATΩ.Z
    nrows, r = size(Y)

    # Compute DAᵀΩ 
    mul!(DAᵀΩ, opN.D, sketch_ATΩ.AᵀΩ)
    # Compute Y = A * DAᵀΩ via mat-vec product avoid receiving linear operator
    for i in 1:r
        @views mul!(Y[:, i], opN.opA, DAᵀΩ[:, i])   
    end

    # Shift for numerical stability
    ν = sqrt(nrows) * eps(norm(Y))
    BLAS.axpy!(ν, Ω, Y) # Y = Y + ν*Ω

    # Obtain eigen decomposition for Nystrom sketch
    mul!(Z, Ω', Y)                          # Compute Z = ΩᵀY = Ωᵀ(ADAᵀΩ)
    chol_fac = cholesky!(Symmetric(Z))      # Cholesky fact: Z = UᵀU
    rdiv!(Y, chol_fac.U)                    # Y = Y * U⁻¹
    @views Nys_sketch.U, Σ, _ = fsvd(Y)     # SVD
    @views Nys_sketch.Λ = Diagonal(max.(0, Σ.^2 .- ν))

    return Nys_sketch
end

##
"""
    mutable struct NysPreconditionerInverse{T}

Mutable version of NystromPreconditionerInverse.
"""
Base.@kwdef mutable struct NysPreconditionerInverse{T}
    N_nys::NystromSketch_mutable{T}
    λmin::T     # smallest eigenvalue of Nystrom sketch
    μ::T        # regularization parameter
    cache::AbstractVector{T}
end

function NysPreconditionerInverse(N_nys::NystromSketch_mutable{T}, μ::T) where T
    if isassigned(N_nys)
        return NysPreconditionerInverse(N_nys, N_nys.Λ.diag[end], μ, zeros(T, rank(N_nys)))
    else
        return NysPreconditionerInverse(N_nys, zero(T), μ, Vector{T}(undef, 0))
    end
end

function NysPreconditionerInverse{T}(initializer::UndefInitializer) where T 
    return NysPreconditionerInverse(NystromSketch_mutable{T}(initializer), 
                                    zero(T), zero(T), Vector{T}(initializer, 0))
end


# Define basic properties
Base.eltype(::NysPreconditionerInverse{T}) where {T} = T
Base.size(P::NysPreconditionerInverse) = (size(P.N_nys.U, 1), size(P.N_nys.U, 1))
Base.size(P::NysPreconditionerInverse, d::Int) = d <= 2 ? size(P)[d] : 1
Base.isassigned(P::NysPreconditionerInverse) = isassigned(P.N_nys)
function Matrix(P::NysPreconditionerInverse)
    return Symmetric(
        (P.λmin + P.μ) * P.N_nys.U*inv(P.N_nys.Λ + P.μ*I)*P.N_nys.U' 
        + (I - P.N_nys.U*P.N_nys.U')
    )
end

function LinearAlgebra.mul!(y, P::NysPreconditionerInverse{T}, x::Vector{T}) where {T <: Real}
    length(y) != length(x) && error(DimensionMismatch())
    
    mul!(P.cache, P.N_nys.U', x)
    @. P.cache *= (P.λmin + P.μ) / (P.N_nys.Λ.diag + P.μ) - one(T)
    mul!(y, P.N_nys.U, P.cache)
    @. y .+= x
    return y
end

function LinearAlgebra.:*(P::NysPreconditionerInverse{T}, x::Vector{T}) where {T <: Real}
    y = similar(x)
    mul!(y, P, x)
    return y
end



"""
    update_NysPreconditionerInverse!(Pinv, Nys_sketch, μ)

Update the Nystrom preconditioner inverse after Nystrom sketch and regularization parameter change.
"""
function update_NysPreconditionerInverse!(Pinv::NysPreconditionerInverse, Nys_sketch::NystromSketch_mutable, μ)
    # # Updata Nystrom sketch
    # @. Pinv.N_nys.U = Nys_sketch.U
    # @. Pinv.N_nys.Λ.diag = Nys_sketch.Λ.diag
    
    # Update λmin and μ
    Pinv.λmin = Nys_sketch.Λ.diag[end]
    Pinv.μ = μ
    
    return Pinv
end


## Helper functions for increasing sketch size
"""
    increase_sketch!(sketch_ATΩ::Sketch_ATΩ{T}, new_rank::Int, opAT::Union{LinearOperator{T}, AdjointLinearOperator{T}})

Increase the number of columns of Ω to new_rank, update AᵀΩ, and reallocate memory for DAᵀΩ, Y, Z. 
"""
function increase_sketch!(sketch_ATΩ::Sketch_ATΩ{T}, new_rank::Int, opAT::Union{LinearOperator{T}, AdjointLinearOperator{T}}) where T
    old_rank = size(sketch_ATΩ.Ω, 2)
    rank_diff = new_rank - old_rank
    @assert rank_diff > 0 "New rank $(new_rank) must be larger than old rank $(old_rank) to increase the sketch."

    ncols, nrows = size(opAT)
    # Get Ω_add such that Ω = [Ω Ω_add]
    Ω_add = inv(sqrt(nrows)) * randn(T, nrows, rank_diff)
    # Compute AᵀΩ_add
    AᵀΩ_add = zeros(T, ncols, rank_diff)
    for i in 1:rank_diff
        @views mul!(AᵀΩ_add[:, i], opAT, Ω_add[:, i])
    end

    # Update Ω and AᵀΩ
    sketch_ATΩ.Ω = isassigned(sketch_ATΩ.Ω) ? hcat(sketch_ATΩ.Ω, Ω_add) : Ω_add
    sketch_ATΩ.AᵀΩ = isassigned(sketch_ATΩ.AᵀΩ) ? hcat(sketch_ATΩ.AᵀΩ, AᵀΩ_add) : AᵀΩ_add

    # Re-allocate DAᵀΩ, Y, Z
    sketch_ATΩ.DAᵀΩ = similar(sketch_ATΩ.AᵀΩ)
    sketch_ATΩ.Y = zeros(T, nrows, new_rank)
    sketch_ATΩ.Z = zeros(T, new_rank, new_rank)

    return nothing
end

"""
    increase_sketch!(Pinv::NysPreconditionerInverse{T}, new_rank::Int)

Reallocate Pinv.cache of size = new_rank for Nystrom preconditioner inverse.
"""
function increase_sketch!(Pinv::NysPreconditionerInverse{T}, new_rank::Int) where T
    Pinv.cache = zeros(T, new_rank)
    return nothing
end


"""
    increase_sketch!(sketch_ATΩ::Sketch_ATΩ{T}, Pinv::NysPreconditionerInverse{T}, new_rank::Int, opAT::Union{LinearOperator{T}, AdjointLinearOperator{T}})

Update sketch_ATΩ and Pinv after increasing sketch size.
"""
function increase_sketch!(sketch_ATΩ::Sketch_ATΩ{T}, Pinv::NysPreconditionerInverse{T}, new_rank::Int, opAT::Union{LinearOperator{T}, AdjointLinearOperator{T}}) where T
    increase_sketch!(sketch_ATΩ, new_rank, opAT)
    increase_sketch!(Pinv, new_rank)
    return nothing
end



"""
Method of preconditioner: Nystrom preconditioner

    struct method_Nystrom

"""
mutable struct method_Nystrom{T}
    sketchsize::Int
    adaptive_sketch::Bool
    max_sketchsize::Union{Int, Nothing}
    sketch_ATΩ::Sketch_ATΩ{T}
    Nys_sketch::NystromSketch_mutable{T}

    function method_Nystrom(sketchsize::Int, adaptive_sketch::Bool; max_sketchsize = nothing, T::DataType = Float64)
        sketch_ATΩ = Sketch_ATΩ{T}(undef)
        Nys_sketch = NystromSketch_mutable{T}(undef)
        return new{T}(sketchsize, adaptive_sketch, max_sketchsize, sketch_ATΩ, Nys_sketch)
    end
end

function allocate_preconditioner(method_P::method_Nystrom{T}, opNreg) where T
    method_P.sketch_ATΩ = Sketch_ATΩ{T}(undef)
    method_P.Nys_sketch = NystromSketch_mutable{T}(undef)
    return NysPreconditionerInverse(method_P.Nys_sketch, zero(T))
end

function update_preconditioner!(method_P::method_Nystrom, Pinv::NysPreconditionerInverse, opN_Reg::opRegNormalEquations, 
                                adaptive_info)
    
    # Update sketchsize if adaptive_sketch is true and adaptive_info is not nothing
    if method_P.adaptive_sketch && !isnothing(adaptive_info)
        update_sketchsize!(method_P, size(opN_Reg, 1), adaptive_info)
    end

    # Get views of cache: sketch_ATΩ and Nys_sketch
    sketch_ATΩ = @views method_P.sketch_ATΩ
    Nys_sketch = @views method_P.Nys_sketch

    # Increase cache if the sketchsize is larger than the current size
    if method_P.sketchsize > size(sketch_ATΩ.Ω, 2)
        increase_sketch!(sketch_ATΩ, Pinv, method_P.sketchsize, opN_Reg.opN.opAT)
    end

    # Update Nystrom preconditioner due to changed D
    update_NystromSketch!(Nys_sketch, sketch_ATΩ, opN_Reg.opN)
    update_NysPreconditionerInverse!(Pinv, Nys_sketch, opN_Reg.δ)

    return nothing
end


function update_sketchsize!(method_P::method_Nystrom, nrow::Int, adaptive_info)

    ## Otherwise (adaptive_sketch = true), decide whether to increase the sketch size.
    # If max_sketchsize not given, set it to be 1/5 of the size of normal equation (i.e. nrow of A).
    isnothing(method_P.max_sketchsize) ? method_P.max_sketchsize = floor(Int, nrow/5) : nothing
    
    # Unwrap adaptive_info
    adaptive_info = adaptive_info[1]    # input adaptive_info is a tuple of Dict
    inneriter = adaptive_info["inneriter"]
    inneriter_jump = adaptive_info["inneriter_jump"]
    
    # Conditions for whether to increase the sketch size.
    conds = Bool[]
    push!(conds, method_P.sketchsize < method_P.max_sketchsize)
    push!(conds, maximum(inneriter_jump ./ max.(1,inneriter)) ≥ 0.5)
    # conds[2] = (conds[2] || maximum(inneriter)/(2*nrow) ≥ 0.06)

    # If all conditions are satisfied, double the sketchsize and cap at max_sketchsize
    if all(conds)
        method_P.sketchsize = min(method_P.max_sketchsize, floor(Int, 2 * method_P.sketchsize))
    end

    return nothing
end