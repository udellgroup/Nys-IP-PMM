######################################################
# NEED TO UNCOMMENT LINES 290-296 IN Nys-IP-PMM.jl TO SAVE DIAGNOSIS DATA
######################################################
using DrWatson
@quickactivate "."
using Random
using Printf


# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("SVM/SVM_run_tests_utils.jl"))

# Set the seed
Random.seed!(1234)

# Load the datanames and methods
methods = [method_Nystrom(200, false)]
problem_name = "CIFAR10_1000"

# Set up the problem type
T = Float64
problem_type = SVMProblem(T)
tol=1e-8

# Run the tests
test_IPPMM(problem_type, problem_name, methods, tol, maxit = 25, timed=false, saved=false);