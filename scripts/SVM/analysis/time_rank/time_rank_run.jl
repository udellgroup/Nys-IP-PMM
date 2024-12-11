using DrWatson
@quickactivate "."
using Random
using Printf

# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("SVM/SVM_run_tests_utils.jl"))

# Set the seed
Random.seed!(123)

# Define methods
ranks = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
methods = vcat(
    [method_Nystrom(rank, false) for rank in ranks], 
    [method_PartialCholesky(rank, true) for rank in ranks]
)
problem_name = "RNASeq"

# Set up the problem type
T = Float64
problem_type = SVMProblem(T)
tol=1e-4

# Run the tests for several rounds
n_rounds = 5
for round in 1:n_rounds
    println("Round $round [out of $n_rounds rounds]")
    
    # Set up the saving directory
    savedir = scriptsdir("SVM", "analysis", "time_rank", "raw_results", problem_name, "Round$(round)")

    # Run the tests
    test_IPPMM(problem_type, problem_name, methods, tol, maxit = 20, timed=true, saved=true, savedir=savedir)

    println("="^80)
end