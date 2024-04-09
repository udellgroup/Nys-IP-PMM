include("normal_equations.jl")

####################################################
#  PartialCholesky
####################################################

"""
    PartialCholesky
"""
mutable struct PartialCholesky{T}
    p::Vector{Integer}  # permutation vector
    H1::AbstractMatrix{T}   # H1 = [H11 H21ᵀ]ᵀ
    chol_H11::Union{Cholesky{T}, SparseArrays.CHOLMOD.Factor{T}}
    DS::Diagonal{T}  # diag(S) = diag(D₂)
    ei::Vector{T}  # cache vector of eᵢ (length m-k) for update_DS!
    tmpk::Vector{T}  # cache vector of length k for update_DS! and mul!
    tmpm::Vector{T}  # cache vector of length m for mul!
    function PartialCholesky(p::Vector{Int}, H1, chol_H11, DS::Diagonal{T}) where {T}
        k = size(chol_H11, 2)
        @assert k <= size(H1, 2) "Cholesky factor of H11 must have same number of columns as H1."
        m = size(H1, 1)
        @assert size(DS, 1) == m - k "Length of DS must be equal to m-k."
        @assert length(p) == m "Length of p must be equal to m."
        return new{T}(p, H1, chol_H11, DS, zeros(T, m-k), zeros(T, k), zeros(T, m))
    end
end

function PartialCholesky(nrow::Int, k::Int; T::DataType = Float64)
    p = collect(1:nrow)
    H1 = zeros(T, nrow, k)
    chol_H11 = cholesky(Diagonal(ones(k)))
    DS = Diagonal(zeros(T, nrow-k))
    return PartialCholesky(p, H1, chol_H11, DS)
end

Base.size(PC::PartialCholesky) = (size(PC.H1, 1), size(PC.H1, 1))
Base.size(PC::PartialCholesky, d::Int) = d <= 2 ? size(PC)[d] : 1
Base.eltype(PC::PartialCholesky) = eltype(PC.H1)

"""
Method of preconditioner: Partial Cholesky preconditioner

    struct method_PartialCholesky{T}

"""
mutable struct method_PartialCholesky{T}
    k_steps::Int
    method_PartialCholesky(k_steps::Int; T::DataType = Float64) = new{T}(k_steps)
end

function allocate_preconditioner(method_P::method_PartialCholesky{T}, opNreg) where T
    return PartialCholesky(size(opNreg, 1), method_P.k_steps; T = T)
end

function update_preconditioner!(method::method_PartialCholesky{S}, 
                                PC::PartialCholesky{S}, 
                                opNreg::opRegNormalEquations, adaptive_info...) where {S}    
    T = eltype(opNreg)
    ncol = LinearAlgebra.checksquare(opNreg)
    k = method.k_steps

    @assert T == S "Input matrix and PartialCholesky must have the same element type."
    @assert size(PC.H1,1) == ncol "Number of rows of PartialCholesky must be equal to the size of the matrix."
    @assert k <= ncol "Rank k has to be smaller than the size of the matrix."
    @assert size(PC.H1,2) == k "Target rank of PartialCholesky must be equal to k."

    # Update diagonal of N
    update_diagN_opN!(opNreg.opN)

    # Sort the diagonal of N in decreasing order
    diagN = @views opNreg.opN.diagN
    p = @views PC.p
    sortperm!(p, diagN, rev=true)
    
    # Form the k columns of N with the largest diagonal elements
    H1 = @views PC.H1
    ei = @views opNreg.opN.ei
    for i in 1:k
        ind = p[i]
        ei[ind] = one(T)
        @views mul!(H1[:, i], opNreg, ei)
        ei[ind] = zero(T)
    end
    @. H1 = H1[p, :]    # permute the rows of H1
    
    # Compute the Cholesky factor of H₁₁
    H11 = Symmetric(@views H1[1:k, 1:k])
    PC.chol_H11 = cholesky(H11)
    
    # Compute the diagonal of H₂₂
    diagH22 = diagN[p[k+1:end]] .+ opNreg.δ   # m-k smaller diagoal elements of N+δI
    
    # Compute diagonal of S
    update_DS!(PC, diagH22)

    return nothing
end

function update_DS!(PC::PartialCholesky{T}, diagH22::Vector{T}) where {T}
    k = size(PC.H1, 2)
    @views diagS = PC.DS.diag
    @views L11 = PC.chol_H11.L
    @views H21 = PC.H1[k+1:end, :]
    @views tmpk = PC.tmpk
    @views ei = PC.ei

    # Make sure PC.H1 and PC.chol_H11 is updated
    # diag(DS) = diag(H₂₂ - H₂₁ * L₁₁⁻ᵀ * L₁₁⁻¹ * H₂₁ᵀ)
    copyto!(diagS, diagH22)
    @inbounds @simd for ind in eachindex(diagS)
        ei[ind] = one(T)
        mul!(tmpk, H21', ei)     # tmpk = H21ᵀ * eᵢ
        ldiv!(L11, tmpk)         # tmpk = L₁₁⁻¹ * tmpk
        diagS[ind] -= dot(tmpk, tmpk)
        ei[ind] = zero(T)
    end
    return nothing
end


function LinearAlgebra.mul!(y, PC::PartialCholesky{T}, x::Vector{T}) where {T <: Real}
    m, k = size(PC.H1)
    length(y) != length(x) && error(DimensionMismatch())
    length(x) == m || error(DimensionMismatch())

    @views H1 = PC.H1
    @views chol_H11 = PC.chol_H11
    @views DS = PC.DS
    @views tmpk = PC.tmpk
    @views tmpm = PC.tmpm
    @views p = PC.p
    @views Ex = x[p]
    @views Ey = y[p]
    @views Ey1 = Ey[1:k]
    @views Ey2 = Ey[k+1:end]
    
    # tmpk = H₁₁⁻¹ (Zᵀ * Ex)
    ldiv!(tmpk, chol_H11, Ex[1:k])    # tmpk = H₁₁⁻¹ * Ex[1:k]

    # Ey = Dₚ⁻¹ * (Ex - H1 * tmpk)
    mul!(Ey, H1, tmpk)                  # Ey = H1 * tmpk
    axpby!(one(T), Ex, -one(T), Ey)     # Ey = Ex - Ey
    ldiv!(DS, Ey2)                      # Ey[k+1:end] = Dₚ⁻¹ * Ey[k+1:end]

    # Ey = Ey + Z * tmpk - Z (H₁₁⁻¹ * H1ᵀ * Ey)
    copyto!(tmpm, Ey)               # Save the value of Ey in tmpm
    axpy!(one(T), tmpk, Ey1)        # compute Ey + Z * tmpk and write in Ey (can overwrite tmpk after this)
    mul!(tmpk, H1', tmpm)           # tmpk = H1ᵀ * tmpm
    ldiv!(chol_H11, tmpk)           # tmpk = H₁₁⁻¹ * tmpk
    axpy!(-one(T), tmpk, Ey1)       # Ey[1:k] .-= tmpk

    return y
end

function LinearAlgebra.:*(PC::PartialCholesky{T}, x::Vector{T}) where {T <: Real}
    y = similar(x)
    mul!(y, PC, x)
    return y
end