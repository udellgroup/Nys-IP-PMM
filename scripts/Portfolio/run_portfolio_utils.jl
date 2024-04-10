using RandomizedPreconditioners

abstract type PortfolioProblem{T} <: AbstractIPMProblem end 

Base.@kwdef struct portfolio_original{T} <: PortfolioProblem{T}
    Σ::AbstractMatrix{T}
    μ::AbstractVector{T}
    B::AbstractMatrix{T}
    Bxub::AbstractVector{T}
end

Base.@kwdef struct portfolio_risk_model{T} <: PortfolioProblem{T}
    Fᵀ::AbstractMatrix{T}   # Factor matrix k × n
    D::Diagonal{T}          # Diagonal matrix n × n
    μ::AbstractVector{T}    # Mean return vector n × 1
    B::AbstractMatrix{T}    # Indices matrix m × n
    Bxub::AbstractVector{T} # Upper bounds for indices m × 1

    function portfolio_risk_model(Fᵀ::AbstractMatrix{T}, D::Diagonal{T}, 
                                  μ::AbstractVector{T}, B::AbstractMatrix{T}, 
                                  Bxub::AbstractVector{T}) where T
        @assert size(Fᵀ, 2) == size(B, 2) "Dimensions of Fᵀ and B do not match."
        @assert size(D, 1) == size(B, 2) "Dimensions of D and B do not match."
        @assert length(μ) == size(B, 2) "Dimensions of μ and B do not match."
        @assert length(Bxub) == size(B, 1) "Dimensions of Bxub and B do not match."
        return new{T}(Fᵀ, D, μ, B, Bxub)
    end
end


function generate_models(m::Int, n::Int, k::Int, d::AbstractVector; T = Float64)
    # Generate Σ
    U = qr(randn(T, n, n)).Q
    Σ = U * Diagonal(d) * U'

    # Generate F and D such that Σ ≈ F * Fᵀ + D
    Σ̂ = NystromSketch(Σ, k, k)
    Fᵀ = sqrt(Σ̂.Λ) * Σ̂.U'
    D = Diagonal(diag(Σ - Fᵀ' * Fᵀ))

    # Generate μ
    μ = randn(n)

    # Generate B and Bxub
    B = zeros(m, n)
    B[1, :] = 5 * randn(n)
    for i in 1:m
        B[i, :] = B[1, :] + randn(n) * 0.001
    end
    Bxub = ones(m) + rand(m)

    return portfolio_risk_model(Fᵀ, D, μ, B, Bxub), portfolio_original(Σ, μ, B, Bxub)
end


##
function get_IPMInput(problem_type::portfolio_risk_model{T}) where T <: Number
    # Extract data
    Fᵀ, D, μ, B, Bxub = problem_type.Fᵀ, problem_type.D, problem_type.μ, problem_type.B, problem_type.Bxub
    
    # Problem dimensions
    m, n = size(B)
    k = size(Fᵀ, 1)

    # Quadratic matrix Q
    diagQ = [D.diag; ones(T, k); spzeros(T, m)]
    opQ = LO.BlockDiagonalOperator(opDiagonal(D.diag), opEye(T, k), opZeros(T, m, m))
    
    # Onjective linear coefficients c
    c = [-μ; zeros(T, k+m)]

    # Constraint matrix A
    op1T = VectorTransposeOperator(ones(T, n))
    opA = [ LO.LinearOperator(B) opZeros(T, m, k) opEye(T, m); 
            op1T opZeros(T, 1, k+m); 
            LO.LinearOperator(Fᵀ) -opEye(T, k) opZeros(T, k, m) ]

    # RHS vector b
    b = [Bxub; one(T); zeros(T, k)]

    # Index Sets and upper bounds xub
    box_ind = Int64[]
    free_ind = collect(n+1:n+k)
    normal_ind = vcat(collect(1:n), collect(n+k+1:n+k+m))
    u = Float64[]

    nrow, ncol = size(opA)
    
    return IPMInput(nrow, ncol, opA, b, c, IPMIndices(normal_ind, box_ind, free_ind), u, opQ, diagQ, :QP)
end

function get_IPPMMParams(problem_type::portfolio_risk_model{T}, tol::T) where T <: Number
    # Initialize IPPMMParams
    params = IPPMMParams(T)
    
    # Use normal equations
    params.normal_eq = true

    # Compute reg_limit via inf norm of A and Q
    inf_norm_A = max((1.0 + norm(problem_type.B, Inf)), 1.0, norm(problem_type.B, Inf) - 1.0)
    inf_norm_Q = maximum(problem_type.D.diag)
    params.reg_limit = max(5 * tol * (1 / max(inf_norm_A^2, inf_norm_Q^2)), 5e-10)
    
    return params
end


Base.@kwdef struct portfolio_diag_approx{T} <: PortfolioProblem{T}
    Σ::AbstractMatrix{T}
    μ::AbstractVector{T}
    B::AbstractMatrix{T}
    Bxub::AbstractVector{T}
    x̄::AbstractVector{T}
    Δ::AbstractVector{T}
end

"""
    Generate portfolio_diag_approx from portfolio_original.
"""
function portfolio_diag_approx(original::portfolio_original{T}, x̄::AbstractVector{T}, Δ::AbstractVector{T}) where T
    return portfolio_diag_approx(original.Σ, original.μ, original.B, original.Bxub, x̄, Δ)
end


function get_IPMInput(problem_type::portfolio_diag_approx{T}) where T <: Number
    # Extract data
    Σ, μ, B = problem_type.Σ, problem_type.μ, problem_type.B
    Bxub, x̄, Δ = problem_type.Bxub, problem_type.x̄, problem_type.Δ
    diagΣ = diag(Σ)

    # Problem dimensions
    m, n = size(B)

    # Quadratic matrix Q
    diagQ = [diagΣ; spzeros(T, n+m)]
    opQ = LO.BlockDiagonalOperator(opDiagonal(diagΣ), opZeros(T, n, n), opZeros(T, m, m))
    
    # Onjective linear coefficients c
    cx = Σ * x̄ - diagΣ .* x̄ - μ
    c = [cx; zeros(T, n+m)]

    # Constraint matrix A
    op1T = VectorTransposeOperator(ones(T, n))
    opA = [ LO.LinearOperator(B) opZeros(T, m, n) opEye(T, m); 
            op1T opZeros(T, 1, n+m); 
            opEye(T, n) -opEye(T, n) opZeros(T, n, m) ]

    # RHS vector b
    b = [Bxub; one(T); x̄ - Δ]

    # Index Sets and upper bounds xub
    box_ind = collect(n+1:2*n)
    free_ind = Int64[]
    normal_ind = vcat(collect(1:n), collect(2*n+1:2*n+m))
    u = 2 * Δ

    nrow, ncol = size(opA)
    
    return IPMInput(nrow, ncol, opA, b, c, IPMIndices(normal_ind, box_ind, free_ind), u, opQ, diagQ, :QP)
end

function get_IPPMMParams(problem_type::portfolio_diag_approx{T}, tol::T) where T <: Number
    # Initialize IPPMMParams
    params = IPPMMParams(T)
    
    # Use normal equations
    params.normal_eq = true

    # Compute reg_limit via inf norm of A and Q
    inf_norm_A = max((1.0 + norm(problem_type.B, Inf)), 1.0)
    inf_norm_Q = maximum(diag(problem_type.Σ))
    params.reg_limit = max(5 * tol * (1 / max(inf_norm_A^2, inf_norm_Q^2)), 5e-10)
    
    return params
end



##
function get_IPMInput_IPPMMParams(problem_type::PortfolioProblem{T}, problem_name::String, tol::T) where T <: Number
    
    # Construct IPMInput
    input = get_IPMInput(problem_type)

    # Construct IPPMMParams
    params = get_IPPMMParams(problem_type, tol)

    return input, params
end


"""
    get_class_name(portfolio::PortfolioProblem)
"""
get_class_name(portfolio::PortfolioProblem) = "Portfolio"

"""
    get_csvnames(time_stamp::String, problem_type::PortfolioProblem, problem_name::String, IPPMM_args::IPPMMargs)
"""
function get_csvnames(time_stamp::String, problem_type::PortfolioProblem, problem_name::String, IPPMM_args::IPPMMargs)
    # Unwrap IPPMM_args
    preconditioner = @views IPPMM_args.preconditioner
    rank = @views IPPMM_args.rank
    tol = @views IPPMM_args.tol

    # Create stem of the filenames
    if typeof(problem_type) <: Union{portfolio_original, portfolio_diag_approx}
        m, n = size(problem_type.B)
        stem = @ntuple ts=time_stamp prob=problem_name m=m n=n pc=preconditioner rank=rank tol=@sprintf("%.e", tol)
    elseif typeof(problem_type) <: portfolio_risk_model
        m, n = size(problem_type.B)
        k = size(problem_type.Fᵀ, 1)
        stem = @ntuple ts=time_stamp prob=problem_name m=m n=n k=k pc=preconditioner rank=rank tol=@sprintf("%.e", tol)
    end
    filestem = savename(stem, sort=false, sigdigits=1)
    
    # Return two filenames ending with _history.csv and _status.csv
    return filestem * "_history.csv", filestem * "_status.csv"
end