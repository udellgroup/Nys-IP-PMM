using DrWatson
@quickactivate "."
using Random
using Printf

# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("Portfolio/run_portfolio_utils.jl"))

# Set the seed
Random.seed!(1234)

# Generate the problem from the risk model
T = Float64
m, n, k = 10000, 50000, 400
d = vcat(collect(range(1e6, 1e1, length=40)), [1 / i for i in 1:(n-40)]) #[1 / i for i in 1:n]
println("Generating the models with m = $m, n = $n, k = $k...")
risk_model, original_model = generate_models(m, n, k, d; T = T)
println("Finished generating the models.")

# Run the IPPMM method on risk model
problem_type = risk_model
problem_name = "risk_model"
tol=1e-8
methods = [method_PartialCholesky(20), method_Nystrom(20, false), method_NoPreconditioner()]
vars = test_IPPMM(problem_type, problem_name, methods, tol, maxit = 40);