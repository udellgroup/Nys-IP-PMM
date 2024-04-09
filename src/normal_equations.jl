####################################################
#  Operator for Normal Equations
####################################################
"""
Operator for Normal Equation of the form
    opA * opD * opAT 
"""
mutable struct opNormalEquations{T}
    opA::LinearOperator{T}
    opAT::AdjointLinearOperator{T} # Aᵀ
    D::Diagonal{T}  # D = (Q + Θₖ⁻¹ + ρₖ Iₘ)⁻¹ or D = I
    sqrtD::Diagonal{T} # √D
    diagN::Vector{T}
    ei::Vector{T}
    tmp::Vector{T}
    function opNormalEquations{T}(opA::LinearOperator{T}, opAT::AdjointLinearOperator{T}, D::Diagonal{T}, 
                                  sqrtD::Diagonal{T}, diagN::Vector{T}, ei::Vector{T}, tmp::Vector{T}) where {T}
        return new{T}(opA, opAT, D, sqrtD, diagN, ei, tmp)
    end
end

"""
opNormalEquations constructor
    
    opNormalEquations(opA::LinearOperator{T})

"""
function opNormalEquations(opA::LinearOperator{T}) where {T}
    nrow, ncol = size(opA)
    D = Diagonal(ones(T, ncol))
    sqrtD = Diagonal(ones(T, ncol))
    opAT = opA'
    diagN = Vector{T}(undef, nrow)
    ei = zeros(T, nrow)
    tmp = Vector{T}(undef, ncol)
    return opNormalEquations{T}(opA, opAT, D, sqrtD, diagN, ei, tmp)
end

function opNormalEquations(A::AbstractMatrix{T}, D::Diagonal{T}) where {T}
    nrow, ncol = size(A)
    opA = LO.LinearOperator(A)
    sqrtD = Diagonal(sqrt.(D.diag))
    opAT = opA'
    diagN = Vector{T}(undef, nrow)
    ei = zeros(T, nrow)
    tmp = Vector{T}(undef, ncol)
    return opNormalEquations{T}(opA, opAT, D, sqrtD, diagN, ei, tmp)
end


function Base.show(io::IO, opN::opNormalEquations)
    println(io, "Operator for Normal Equations of the form")
    println(io, "    opA * D * opAT")
    println(io, "where opA is a LinearOperator of size $(size(opN.opA)) and D is a Diagonal matrix.")
end


"""
Define mul! and * for opNormalEquations
"""
function LinearAlgebra.mul!(y::AbstractArray{T}, opN::opNormalEquations{T}, x::AbstractArray{T}) where {T}
    # @assert size(y) == size(x) "Vectors x and y must have same dimension."
    tmp_ncol = @views opN.tmp
    mul!(tmp_ncol, opN.opAT, x)
    mul!(tmp_ncol, opN.D, tmp_ncol)
    mul!(y, opN.opA, tmp_ncol)
    return y
end

LinearAlgebra.mul!(opN::opNormalEquations{T}, x::AbstractVector{T}) where {T} = mul!(x, opN, x)


function LinearAlgebra.:*(opN::opNormalEquations{T}, x::AbstractArray{T}) where {T}
    y = similar(x)
    mul!(y, opN, x)
    return y
end

##
"""
    update_diagN_opN!(opN)
"""
function update_diagN_opN!(opN::opNormalEquations{T}) where {T}
    @views diagN = opN.diagN
    @views opAT = opN.opAT
    @views tmp = opN.tmp
    @views ei = opN.ei

    # Make sure opN.sqrtD is updated
    @. opN.sqrtD.diag = sqrt(opN.D.diag)
    @inbounds @simd for ind in eachindex(diagN)
        ei[ind] = one(T)
        mul!(tmp, opAT, ei)
        lmul!(opN.sqrtD, tmp)
        diagN[ind] = dot(tmp, tmp)
        ei[ind] = zero(T)
    end
    return nothing
end

function LinearAlgebra.diag(opN::opNormalEquations{T}) where {T}
    return opN.diagN
end

##
function Base.size(opN::opNormalEquations)
    return opN.opA.nrow, opN.opA.nrow
end

function Base.size(opN::opNormalEquations, dim::Integer)
    if dim == 1
        return opN.opA.nrow
    elseif dim == 2
        return opN.opA.nrow
    else
        throw(DimensionMismatch("Dimension must be 1 or 2"))
    end
end

function Base.eltype(opN::opNormalEquations)
    return eltype(opN.opA)
end



##
"""
    compute_diagD!(diagD, diagQ, ρ, vars, indices)
"""
function update_opN!(opN, diagQ, ρ, vars::IPMVariables, indices::IPMIndices)
    
    diagD = @views opN.D.diag
    # Views of IPM Variables
    x = @views vars.x
    z = @views vars.z
    w = @views vars.w
    s = @views vars.s

    # Indices of the problem
    ind_posunbdd = @views indices.normal
    ind_box = @views indices.box
    ind_free = @views indices.free
    n_IJ = length(ind_posunbdd) + length(ind_box)

    copyto!(diagD, diagQ) # diagD = Q_bar
    # diagD = (Q + Θₖ⁻¹ + ρₖ Iₘ)
    if (n_IJ > 0)
        @. diagD[ind_posunbdd] += z[ind_posunbdd] / x[ind_posunbdd] + ρ
        @. diagD[ind_box] += z[ind_box] / x[ind_box] + s[ind_box] / w[ind_box] + ρ
        @. diagD[ind_free] += ρ
    else
        @. diagD += ρ
    end
    # diagD = (Q + Θₖ⁻¹ + ρₖ Iₘ)⁻¹
    @. diagD = inv(diagD)
    @. opN.sqrtD.diag = sqrt(diagD)
    return nothing
end


##
"""
    Regularized Normal Equations
"""


mutable struct opRegNormalEquations{T}
    opN::opNormalEquations{T}
    δ::T # Regularized Parameter 
    function opRegNormalEquations{T}(opN::opNormalEquations{T}, δ::T) where {T}
        return new{T}(opN, δ)
    end
end

opRegNormalEquations(opN::opNormalEquations{T}, δ::T) where {T} = opRegNormalEquations{T}(opN, δ)

opRegNormalEquations(opN::opNormalEquations{T}) where {T} = opRegNormalEquations{T}(opN, zero(T))


function opRegNormalEquations(opA::LinearOperator{T}, δ::T) where {T}
    opN = opNormalEquations(opA)
    return opRegNormalEquations(opN, δ)    
end

function Base.show(io::IO, opNreg::opRegNormalEquations)
    println(io, "Operator for Regularized Normal Equations of the form")
    println(io, "    opA * D * opAT + δ * I")
    println(io, "where opA is a LinearOperator of size $(size(opNreg.opN.opA)) and D is a Diagonal matrix.")
end

function LinearAlgebra.mul!(y::AbstractArray{T}, opNreg::opRegNormalEquations{T}, x::AbstractArray{T}) where {T}
    @assert size(y) == size(x) "Vectors x and y must have same dimension."
    mul!(y, opNreg.opN, x)
    @. y += opNreg.δ * x
    return y
end

LinearAlgebra.mul!(opNreg::opRegNormalEquations{T}, x::AbstractArray{T}) where {T} = mul!(x, opNreg, x)
   

function LinearAlgebra.:*(opNreg::opRegNormalEquations{T}, x::Vector{T}) where {T}
    y = similar(x)
    mul!(y, opNreg, x)
    return y
end


function LinearAlgebra.diag(opNreg::opRegNormalEquations{T}) where {T}
    return opNreg.opN.diagN .+ opNreg.δ
end

Base.size(opNreg::opRegNormalEquations) = size(opNreg.opN)
Base.size(opNreg::opRegNormalEquations, dim::Integer) = size(opNreg.opN, dim)
Base.eltype(opNreg::opRegNormalEquations) = eltype(opNreg.opN)


##
"""
    update_opN_Reg!(opN)
"""
function update_opN_Reg!(opNreg::opRegNormalEquations, diagQ, ρ, δ, vars::IPMVariables, indices::IPMIndices)
    update_opN!(opNreg.opN, diagQ, ρ, vars, indices)
    opNreg.δ = δ
    return nothing
end


### Normal Equations solver
"""
    normal_eq_solve!

Normal equation solver
"""
function normal_eq_solve!(steps::IPMVariables{T}, cg_solver, 
                         opN_Reg, Pinv,
                         vars::IPMVariables{T}, res::IPMResiduals{T}, indices::IPMIndices,
                         krylov_tol::Number) where {T<:Number}
    m, n = length(vars.y), length(vars.x)
    instability = false
    x = @views vars.x
    z = @views vars.z
    w = @views vars.w
    s = @views vars.s
    dx = @views steps.x
    dy = @views steps.y
    dz = @views steps.z
    dw = @views steps.w
    ds = @views steps.s

    res_p = @views res.primal
    res_d = @views res.dual
    res_u = @views res.upper
    res_xz = @views res.compl_xz
    res_ws = @views res.compl_ws

    ind_posunbdd = @views indices.normal
    ind_box = @views indices.box
    ind_free = @views indices.free
    n_IJ = length(ind_posunbdd) + length(ind_box)

    if n_IJ > 0
        aug_rhsₓ = zeros(T, n)
    end
    rhs = @views opN_Reg.opN.ei # Uses ei as temporary rhs
    tmp_col = @views opN_Reg.opN.tmp 

    if n_IJ > 0
        @views @. aug_rhsₓ[ind_posunbdd] = -(res_xz[ind_posunbdd] / x[ind_posunbdd])
        @views @. aug_rhsₓ[ind_box] = -(res_xz[ind_box] / x[ind_box]) + (res_ws[ind_box] / w[ind_box]) - (res_u * s[ind_box] / w[ind_box])
        @views @. aug_rhsₓ += res_d
        # rhs = res_p + opA * (D * aug_rhsₓ)
        mul!(tmp_col, opN_Reg.opN.D, aug_rhsₓ)
    else
        # rhs = res_p + opA * (D * res_d)
        mul!(tmp_col, opN_Reg.opN.D, res_d)
    end
    mul!(rhs, opN_Reg.opN.opA, tmp_col)
    @. rhs += res_p
    

    ## CG solver
    cg!(cg_solver, opN_Reg, rhs, M=Pinv, rtol=krylov_tol) #Change rtol to check convergence!
    copyto!(dy, cg_solver.x)

    # dx = D*(Aᵀ * dy - aug_rhsₓ) or dx = D*(Aᵀ * dy - res_d)
    mul!(dx, opN_Reg.opN.opAT, dy)
    if n_IJ > 0
        @. dx -= aug_rhsₓ
    else
        @. dx -= res_d
    end
    mul!(dx, opN_Reg.opN.D, dx)

    if n_IJ > 0
        @views @. dw[ind_box] = res_u - dx[ind_box]
        @views @. dw[ind_posunbdd] = zero(T)
        @views @. dw[ind_free] = zero(T)
        @views @. dz[ind_posunbdd] = (res_xz[ind_posunbdd] - z[ind_posunbdd] * dx[ind_posunbdd]) / x[ind_posunbdd]
        @views @. dz[ind_box] = (res_xz[ind_box] - z[ind_box] * dx[ind_box]) / x[ind_box]
        @views @. dz[ind_free] = zero(T)
        @views @. ds[ind_box] = (res_ws[ind_box] - s[ind_box] * dw[ind_box]) / w[ind_box]
        @views @. ds[ind_posunbdd] = zero(T)
        @views @. ds[ind_free] = zero(T)
    end

    # Makes sure that ei = 0.0 when leaving
    fill!(rhs, zero(T))
    return instability, cg_solver.stats.niter
end