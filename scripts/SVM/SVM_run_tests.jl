using DrWatson
@quickactivate "."
using Random
using Printf

using JLD2

# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("SVM/SVM_run_tests_utils.jl"))

# Set the seed
Random.seed!(1234)

# Load the datanames and methods
include(scriptsdir("SVM/SVM_run_tests_data2methods.jl"))
problem_names_list = keys(data2methods)
# problem_names_list = ["CIFAR10_1000"]

# Set up the problem type
T = Float64
problem_type = SVMProblem(T)

# Run the tests
tol=1e-4
for problem_name in problem_names_list
    test_IPPMM(problem_type, problem_name, data2methods[problem_name]["method_Ps"], tol, maxit = 25, timed=false, saved=true);
end