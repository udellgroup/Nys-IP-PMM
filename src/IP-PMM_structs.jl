# Quadratic Porgram 

# minimize   1/2 * xᵀQx + cᵀx
# subject to Ax = b
#            xF: free variables
#            xᴵ ≥ 0
#            0 ≤ xᴶ ≤ u

"""
    Structure of IPM variables
"""
mutable struct IPMVariables{T}
    x::AbstractVector{T}    # Primal variables
    y::AbstractVector{T}    # Free dual variables
    z::AbstractVector{T}    # Complementarity dual variables for xᴵ ≥ 0, xᴶ ≥ 0
    w::AbstractVector{T}    # Slack primal variables with wᴶ = u - xᴶ
    s::AbstractVector{T}    # Complementarity dual variables for xᴶ ≤ u (or w ≥ 0)

    function IPMVariables(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, w::AbstractVector{T}, s::AbstractVector{T}) where T
        @assert length(x) == length(z) "Primal variables x and complementarity dual variables z must have same length."
        @assert length(w) == length(s) "Primal variables w and complementarity dual variables s must have same length."
        return new{T}(x, y, z, w, s)
    end
end

"""
IPMVariables constructor

    IPMVariables(m, n_pos; n_free = 0, n_box = 0, T = Float64)

m: number of rows of A
n_pos: number of positive variables = |I ∪ J|
n_free: number of free variables = |F|
n_box: number of bounded variables = |J|
"""
function IPMVariables(m::Int, n_pos::Int; n_free::Int = 0, n_box::Int = 0, T::Type = Float64)
    @assert n_pos ≥ n_box "Number of positive variables must be greater than or equal to number of bounded variables."
    n = n_pos + n_free      # number of all variables
    x = Vector{T}(undef, n)        
    y = Vector{T}(undef, m)
    z = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    s = Vector{T}(undef, n)
    return IPMVariables(x, y, z, w, s)
end
IPMVariables{T}(m::Int, n_pos::Int) where T = IPMVariables(m, n_pos; T = T)

"""
isempty(), copyto!() for IPMVariables
"""
Base.isempty(vars::IPMVariables) = (isempty(vars.x) || isempty(vars.y) || isempty(vars.z) || isempty(vars.w) || isempty(vars.s))
function Base.copyto!(dest::IPMVariables, src::IPMVariables)
    copyto!(dest.x, src.x)
    copyto!(dest.y, src.y)
    copyto!(dest.z, src.z)
    copyto!(dest.w, src.w)
    copyto!(dest.s, src.s)
    return dest
end

"""
    Structure of IPM indices
"""
Base.@kwdef struct IPMIndices{Int}
    normal::AbstractArray{Int}      # Indices of positive unbounded variables xᴵ ≥ 0
    box::AbstractArray{Int}         # Indices of box-constrained variables 0 ≤ xᴶ ≤ u
    free::AbstractArray{Int}        # Indices of free variables
end

"""
    IPMIndices constructor

    IPMIndices(ncol::Int, xlb::AbstractVector, xub::AbstractVector)

- ncol: number of columns of constraint matrix A (= number of variables x)
- xlb: lower bounds of variables x
- xub: upper bounds of variables x
"""
function IPMIndices(ncol::Int, xlb::AbstractVector, xub::AbstractVector)
    free_ind = findall(xlb .== -Inf .&& xub .== Inf)
    box_ind = findall(xlb .== 0.0 .&& xub .< Inf)
    normal_ind = setdiff([i for i in 1:ncol], vcat(free_ind, box_ind))
    return IPMIndices(normal_ind, box_ind, free_ind)
end


"""
    Structure of IPM residuals
"""
Base.@kwdef mutable struct IPMResiduals{T}
    primal::AbstractVector{T}       # Primal residual: b - Axₖ
    dual::AbstractVector{T}         # Dual residual: c - Aᵀyₖ - zₖ + Qxₖ
    upper::AbstractVector{T}        # Upper bound residual: u - xₖ - wₖ
    compl_xz::AbstractVector{T}     # Complementarity residual: σₖμₖeₙ - XₖZₖeₙ
    compl_ws::AbstractVector{T}     # Complementarity residual: σₖμₖeₙ - WₖSₖeₙ
end

"""
IPMResiduals constructor

    IPMResiduals(m::Int, n:Int, n_J::Int, n_IJ::Int; T::Type = Float64)

m: number of rows of A
n: number of all variables
n_J: number of box-constrained variables = |J|
n_IJ: number of non-negative variables = |I ∪ J|
"""
function IPMResiduals(m::Int, n::Int, n_J::Int, n_IJ::Int; T::Type = Float64)
    r_p = Vector{T}(undef, m)
    r_d = Vector{T}(undef, n)
    r_u = Vector{T}(undef, n_J)
    r_xz = Vector{T}(undef, n)
    r_ws = Vector{T}(undef, n)
    return IPMResiduals(r_p, r_d, r_u, r_xz, r_ws)
end
IPMResiduals{T}(m::Int, n::Int, n_J::Int, n_IJ::Int) where T = IPMResiduals(m, n, n_J, n_IJ; T = T)
IPMResiduals(m::Int, indices::IPMIndices) = IPMResiduals(m, length(indices.normal) + length(indices.box) + length(indices.free), length(indices.box), length(indices.normal) + length(indices.box))
IPMResiduals{T}(m::Int, indices::IPMIndices) where T = IPMResiduals(m, length(indices.normal) + length(indices.box) + length(indices.free), length(indices.box), length(indices.normal) + length(indices.box); T = T)


"""
    Structure of IPM input
"""
struct IPMInput{T}
    nrow::Int                      # Number of rows of constraint matrix A
    ncol::Int                      # Number of columns of constraint matrix A
    opA::LO.LinearOperator{T}      # Linear operator of constraint matrix A
    b::AbstractVector{T}           # Constraint RHS vector b
    c::AbstractVector{T}           # Objective linear coefficients c
    indices::IPMIndices{Int}       # Indices of normal vars xᴵ ≥ 0, box vars 0 ≤ xᴶ ≤ u, and free vars xF
    u::AbstractVector{T}           # Upper bounds of box-constrained vars
    opQ::LO.LinearOperator{T}      # Linear operator of quadratic matrix Q
    diagQ::AbstractVector{T}       # Diagonal of quadratic matrix Q
    prob_type::Symbol              # Type of problem: :LP, :QP

    function IPMInput(nrow::Int, ncol::Int,
                      opA::LO.LinearOperator{T}, b::AbstractVector{T}, c::AbstractVector{T}, 
                      indices::IPMIndices{Int}, u::AbstractVector{T}, 
                      opQ::LO.LinearOperator{T}, diagQ::AbstractVector{T}, prob_type::Symbol) where T
        @assert length(c) == ncol "Vector c must have same length as ncol."
        @assert length(b) == nrow "Vector b must have same length as nrow."
        @assert length(u) == length(indices.box) "Upper bounds u must have same length as number of box vars."
        @assert prob_type == :LP || prob_type == :QP "Problem type must be either :LP or :QP."
        if prob_type == :QP
            @assert size(opQ, 1) == size(opQ, 2) == ncol "Q must be square and has dimension ncol."
            @assert length(diagQ) == ncol "diagQ must have same length as dimension of Q."
        end
        return new{T}(nrow, ncol, opA, b, c, indices, u, opQ, diagQ, prob_type)
    end
end

"""
    IPMInput constructor for stanford format LP mps data
"""
function IPMInput(nrow::Int, ncol::Int, 
                  c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T},
                  xlb::AbstractVector{T}, xub::AbstractVector{T}) where T
    # Make sure b and c are dense vectors
    b = Vector(b)
    c = Vector(c)

    # Put matrix A into LO.LinearOperator and set opQ and diagQ to be zero
    opA = LO.LinearOperator(A)
    opQ = opZeros(T, ncol, ncol)
    diagQ = spzeros(T, ncol)

    # Construct indices
    indices = IPMIndices(ncol::Int, xlb::AbstractVector, xub::AbstractVector)

    # Construct upper bounds
    u = xub[indices.box]

    return IPMInput(nrow, ncol, opA, b, c, indices, u, opQ, diagQ, :LP)
end


"""
    IPMInput constructor for stanford format diagonal QP mps data
"""
function IPMInput(nrow::Int, ncol::Int,
    c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T},
    xlb::AbstractVector{T}, xub::AbstractVector{T}, diagQ::AbstractVector{T}) where {T}
    # Make sure b and c are dense vectors
    b = Vector(b)
    c = Vector(c)

    # Put matrix A into LO.LinearOperator and set opQ and diagQ to be zero
    opA = LO.LinearOperator(A)
    opQ = opDiagonal(ncol, ncol, diagQ)

    # Construct indices
    indices = IPMIndices(ncol::Int, xlb::AbstractVector, xub::AbstractVector)

    # Construct upper bounds
    u = xub[indices.box]

    return IPMInput(nrow, ncol, opA, b, c, indices, u, opQ, diagQ, :QP)
end


"""
    Structure of IPM parameters
"""
Base.@kwdef mutable struct IPPMMParams{T}
    normal_eq::Bool
    reg_limit::T    # Controlled perturbation.
    krylov_tol::T
    max_sketchsize::Int
end

"""
IPPMMParams constructor

    IPPMMParams(T::DataType; adaptive_sketch::Bool, krylov_tol::Real)
    IPPMMParams(A::AbstractMatrix{T}, tol::Real)
"""
function IPPMMParams(T::DataType)
    normal_eq = false
    krylov_tol = convert(T, 1e-6)
    reg_limit = convert(T, 1e-10)
    max_sketchsize = -1
    return IPPMMParams{T}(normal_eq, reg_limit, krylov_tol, max_sketchsize)
end

function IPPMMParams(A::AbstractMatrix{T}, tol::Real) where T
    normal_eq = false
    reg_limit = convert(T, max(tol/(norm(A, Inf)^2), 1e-10))
    krylov_tol = convert(T, 1e-6)
    max_sketchsize = -1
    return IPPMMParams{T}(normal_eq, reg_limit, krylov_tol, max_sketchsize)
end
