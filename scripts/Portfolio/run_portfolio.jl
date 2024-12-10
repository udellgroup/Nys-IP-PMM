using DrWatson
@quickactivate "."
using LinearAlgebra
using Random
using Printf

# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("Portfolio/run_portfolio_utils.jl"))

# Set the seed
Random.seed!(1234)

# Set the parameters
T = Float64
m, n, k = 50000, 80000, 100
isolated_num = 20
d_fast = [10^i for i in range(16, 1, length=isolated_num)]
d_slow = [1 / i for i in 1:(n-isolated_num)]
d = vcat(d_fast, d_slow)

# Generate the problem or load it from the file
risk_model, original_model = generate_models(m, n, k, d; T = T, saved = true)

# Run the IPPMM method on risk model
problem_type = risk_model
problem_name = "risk_model"
tol=1e-8
methods = [method_Nystrom(20, false), method_NoPreconditioner(), method_PartialCholesky(20)]
vars = test_IPPMM(problem_type, problem_name, methods, tol, maxit = 40, init_Pinv = I);