## This script computes the nnz ratio of L21 for Partial Cholesky preconditioner at each IP-PMM iteration.
## Problem: SVM problem with CIFAR10_1000 dataset 

using DrWatson
@quickactivate "."
using LinearAlgebra
using Random
using BenchmarkTools

# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("SVM/SVM_run_tests_utils.jl"))

# Inputs
T = Float64
problem_type = SVMProblem(T)
problem_name = "CIFAR10_1000"
total_iters = 15    # Total number of iterations (depends on the problem_name)
tol = 1e-8

# Construct A and setup opA
X, y = load_data(problem_type, problem_name)
n, d = size(X)
Id = Diagonal(ones(n))
A = [Id -X * Diagonal(y); zeros(1, n) y']
opA = LinearOperator(A)
opAT = opA'
T = eltype(opA)
nrows, ncols = size(A)

# Directory for diagD and δ
filedir = scriptsdir("SVM", "analysis", "eigs_dist", "results_diagD")

# Preallocate a vector to store sparsity ratios
nnz_ratios = zeros(total_iters)

# Loop over all iterations
for iter in 1:total_iters
    # Load diagD and δ for the current iteration
    diagD_filename = "diagD_tol=$(tol)_iter=$(iter).jld2"
    filepath = joinpath(filedir, diagD_filename)
    diagD, δ = load(filepath, "diagD", "delta")

    # Compute N = A * D * A' and setup opN
    D = Diagonal(diagD)
    N = Symmetric(A * D * A')
    opN = opNormalEquations(opA)
    opNreg = opRegNormalEquations(opN, δ)

    ## Compute Partial Cholesky preconditioner
    rk = 200
    method = method_PartialCholesky(rk)
    PC = allocate_preconditioner(method, opNreg)
    update_preconditioner!(method, PC, opNreg, A)

    # Compute L₂₁
    L21 = zeros(T, nrows-rk, rk)
    copyto!(L21, PC.H1[rk+1:end, 1:rk])
    rdiv!(L21, PC.chol_H11.L)

    # Count nonzeros in L21 (threshold: 1e-4)
    nnz_ratio = sum(abs.(L21) .> 1e-2) / (rk * (nrows - rk))
    nnz_ratios[iter] = nnz_ratio

    # Display nnz_ratio for the current iteration
    println("Iteration $iter: nnz_ratio of L21 = $nnz_ratio")
end
