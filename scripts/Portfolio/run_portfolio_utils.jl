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
    row1 = 5 * randn(1,n)
    B = repeat(row1, m, 1)
    B[2:m, :] += randn(m-1, n) * 0.001
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
    if typeof(problem_type) <: portfolio_original
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