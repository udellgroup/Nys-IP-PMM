using DataFrames, CSV, Dates
using Random
using SparseArrays

"""
    SVMProblem(X, y, τ)

Construct a SVMProblem object with feature matrix X, label vector y, and penalty parameter τ.
One possible choice for argument "problem_type" in function test_IPPMM().
"""
Base.@kwdef mutable struct SVMProblem{T} <: AbstractIPMProblem
    X::AbstractMatrix{T}    # feature matrix
    y::AbstractVector{T}    # label vector
    τ::T    # penalty parameter for misclassifications (default: 1.0)
end
SVMProblem(T::DataType) = SVMProblem{T}(Array{T}(undef, 0, 0), Vector{T}(undef, 0), one(T))
Base.eltype(::SVMProblem{T}) where T = T

"""
    get_IPMInput(problem_type::SVMProblem, problem_name::String)

Given SVMProblem object {problem_type} and dataset {problem_name}, construct the corresponding Linearoperators in standard QP form for dual SVM QP.
"""
function get_IPMInput_IPPMMParams(problem_type::SVMProblem, problem_name::String, tol)
    # Load data: features and labels
    println("Loading data for ", problem_name, "...")
    problem_type.X, problem_type.y = load_data(problem_type, problem_name)
    println("Successfully loaded ", problem_name, ". Number of features: ", size(problem_type.X, 1))
    
    # Construct IPMInput
    input =  SVMdata2dualQP(problem_type.X, problem_type.y, problem_type.τ)

    # Construct IPPMMParams
    params = get_IPPMMParams(problem_type, problem_type.X, problem_type.y, tol)

    return input, params
end

"""
    get_IPPMMParams(problem_type::SVMProblem, X::AbstractArray{T}, y::AbstractVector{T}, tol::T)

Update parameters of IP-PMM for SVM problem.
"""
function get_IPPMMParams(problem_type::SVMProblem, X::AbstractArray{T}, y::AbstractVector{T}, tol::T) where T <: Number
    # Initialize IPPMMParams
    params = IPPMMParams(T)
    
    # Always use normal equations for SVM problem
    params.normal_eq = true

    # Compute reg_limit via inf norm of A and Q
    XY = X * Diagonal(y);
    inf_norm_A = max((1.0 + norm(XY, Inf)), sum(abs.(y)))
    inf_norm_Q = 1.0
    params.reg_limit = max(5 * tol * (1 / max(inf_norm_A^2, inf_norm_Q^2)), 5e-10)
    
    return params
end

"""
    get_class_name(svm::SVMProblem)

Return problem class string "SVM" for SVMProblem object. Used for constructing saving directory.
"""
get_class_name(svm::SVMProblem) = "SVM"


"""
    get_csvnames(time_stamp::String, problem_type::SVMProblem, problem_name::String, IPPMM_args::IPPMMargs)
"""
function get_csvnames(time_stamp::String, problem_type::SVMProblem, problem_name::String, IPPMM_args::IPPMMargs)
    # Unwrap IPPMM_args
    preconditioner = @views IPPMM_args.preconditioner
    rank = @views IPPMM_args.rank
    tol = @views IPPMM_args.tol

    # Create stem of the filenames
    stem = @ntuple ts=time_stamp prob=problem_name pc=preconditioner rank=rank tol=@sprintf("%.e", tol)
    filestem = savename(stem, sort=false, sigdigits=1)
    
    # Return two filenames ending with _history.csv and _status.csv
    return filestem * "_history.csv", filestem * "_status.csv"
end

## 
"""
    load_data(problem_type::SVMProblem, problem_name::String)

Load dataset for SVMProblem with name problem_name.
"""
function load_data(problem_type::SVMProblem, problem_name::String)
    if problem_name in ["CIFAR10_1000", "sector", "dexter", "arcene", "RNASeq", "STL10", "CIFAR10", "SensIT", "SVHN"]
        include(scriptsdir("SVM/data_preprocess/SVM_preproc_" * problem_name * ".jl"))
        return X, y
    elseif problem_name in keys(SVM_DATA_FILES)
        # Set random features parameters if available
        rf_params = haskey(SVM_RAND_FEAT_PARAMS, problem_name) ? SVM_RAND_FEAT_PARAMS[problem_name] : nothing
        return load_SVM_preprocessed_data(problem_name, rf_params=rf_params)
    elseif problem_name in keys(OPENML_DATA_INFO)
        # Set random features parameters if available
        rf_params = haskey(OPENML_RAND_FEAT_PARAMS, problem_name) ? OPENML_RAND_FEAT_PARAMS[problem_name] : nothing
        return load_openml_preprocess_data(OPENML_DATA_INFO[problem_name], rf_params=rf_params)
    end
end

"""
    SVMdata2dualQP(X, y, τ)

Given m × n data matrix X and label vector y ∈ Rⁿ, construct the corresponding Linearoperators in standard QP form for dual SVM QP. Return also three sets of indices for box variables, normal variables, and free variables.

m: number of features
n: number of samples

Dual SVM QP (variables: w ∈ Rᵐ, p ∈ Rⁿ):
||               min  1/2 wᵀw - 1ᵀp                                 ||
||               s.t. w - XYp = 0                                   ||
||                        yᵀp = 0                                   ||
||                    w free                                        ||
||                    0 ≤ p ≤ τ                                     ||

Standard QP form:
||               min  1/2 xᵀQx + cᵀx                                ||
||               s.t. A x   = b          x - primal variable        ||
||                      0 <= x <= u      "box"    variables         ||
||                      0 <= x           "normal" variables         ||
||                           x free      "free"   variables         ||

 Q = [ Iₘ  0 ]      c = [  0ₘ ]
     [ 0   0 ]          [ -1ₙ ]

A = [ Iₘ  -XY ]      b = [ 0ₘ ]
    [ 0    yᵀ ]          [ 0  ]

free: 1:m   box: m+1:m+n 
"""
function SVMdata2dualQP(X::AbstractMatrix{T}, y::AbstractVector{T}, τ::T) where T <: Number
    m, n = size(X)
    @assert length(y) == n "Number of samples must be equal to length of label vector y."

    # Quadratic matrix Q
    diagQ = [ones(T, m); spzeros(T, n)]
    opQ = LO.BlockDiagonalOperator(opEye(T, m), opZeros(T, n, n))
    
    # Onjective linear coefficients c
    c = [zeros(T, m); -ones(T, n)]

    # Constraint matrix A
    XY = X * Diagonal(y)
    opminusXY = LO.LinearOperator(-XY)
    opyT = VectorTransposeOperator(y)
    opA = [opEye(T, m) opminusXY ; opZeros(T, 1, m) opyT]

    # RHS vector b
    b = zeros(T, m + 1)

    # Index Sets and upper bounds xub
    box_ind = collect(m+1:m+n)
    free_ind = collect(1:m)
    normal_ind = Int64[]
    u = τ * ones(T, length(box_ind))

    nrow, ncol = size(opA)
    
    return IPMInput(nrow, ncol, opA, b, c, IPMIndices(normal_ind, box_ind, free_ind), u, opQ, diagQ, :QP)
end

function get_A_matrix(problem_type::SVMProblem, problem_name::String)
    @views X = problem_type.X
    @views y = problem_type.y
    m = size(X, 1)

    # Constraint matrix A
    XY = X * Diagonal(y)
    A = [ I -XY; spzeros(1, m) y']
        
    return A
end