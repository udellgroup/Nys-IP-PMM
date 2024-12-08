using LinearAlgebra
using Plots
using Measures
using LaTeXStrings
using JLD2  # Ensure you can load JLD2 files

include(scriptsdir("SVM/SVM_run_tests_utils.jl"))

# Inputs
T = Float64
problem_type = SVMProblem(T)
problem_name = "CIFAR10_1000"
total_iters = 15    # Total number of iterations (depends on the problem_name)

# Construct A
X, y = load_data(problem_type, problem_name)
n, d = size(X)
Id = Diagonal(ones(n))
A = [Id -X * Diagonal(y); zeros(1, n) y']

# Tolerance
tol = 1e-8

# Initialize storage for effective dimensions
effective_dimensions = zeros(total_iters)
eigvals_N = zeros(total_iters, n+1)

# Loop through iterations
for iter in 1:total_iters
    @printf("Handling iteration %d\n", iter)
    
    # Load diagD and ρ
    filedir = scriptsdir("SVM", "analysis", "eigs_dist", "results_diagD")
    diagD_filename = "diagD_tol=$(tol)_iter=$(iter).jld2"
    filepath = joinpath(filedir, diagD_filename)
    diagD, δ = load(filepath, "diagD", "delta")
    D = Diagonal(diagD)
    
    # Compute N = A * D * A'
    N = Symmetric(A * D * A')
    
    # Compute eigenvalues of N
    eigvals_N[iter, :] = eigvals(N)
    eigvals_N[iter, :] = sort(eigvals_N[iter, :], rev=true)
    
    # Compute effective dimension
    effective_dimensions[iter] = sum(eigvals_N[iter, :] ./ (eigvals_N[iter, :] .+ δ))
end

# Compute eigvals of AAT
AAT = A * A'
eigvals_AAT = eigvals(AAT)
eigvals_AAT = sort(eigvals_AAT, rev=true)


## Plotting
plot_font = "Computer Modern"
default(fontfamily=plot_font)

# Plot effective dimension over iterations
plot(iterations, effective_dimensions, seriestype = :line, lw = 2, marker=(:circle,5),
    xlabel = "IP-PMM iteration", ylabel = "Effective Dimension", 
    title = "Effective dimension", legend = false)
savefig(joinpath(@__DIR__, "../..", "plots", "SVM", "CIFAR10_1000", "eff_dim.pdf"))

# Plot eigenvalues of AAT
plot(eigvals_AAT, seriestype = :scatter, yscale = :log10, 
            xlabel = "Index", ylabel = "Eigenvalues", 
            title = latexstring("Eigenvalues of \$AA^T\$"), 
            legend = false, markersize = 2, markerstrokewidth = 0)
savefig(joinpath(@__DIR__, "../..", "plots", "SVM", "CIFAR10_1000", "EigenvaluesAAT.pdf"))


# Plot eigenvalues of AAT and at 2 different iterations
plot_iters = [12, 15]
global_ylim_log = log10.(extrema(vcat(eigvals_N[plot_iters, :])))
global_ylim = (10^floor(global_ylim_log[1]-1), 10^ceil(global_ylim_log[2]+1))
plt = plot(layout=(1, 3), size=(1200, 400), margin=10mm)  # 1 row, 3 columns
plot!(plt, eigvals_AAT, seriestype=:scatter, yscale=:log10, 
        xlabel="Index", ylabel="Eigenvalues",
        title=latexstring("\$AA^T\$"), 
        legend=false, markersize=2, markerstrokewidth=0, subplot=1,
        ylim=global_ylim)
for (i, plt_iter) in enumerate(plot_iters)
    plot!(plt, eigvals_N[plt_iter, :], seriestype=:scatter, yscale=:log10, 
        xlabel="Index", 
        # ylabel=i == 1 ? "Eigenvalues" : "",
        title="Iteration $(plt_iter)", 
        legend=false, markersize=2, markerstrokewidth=0, subplot=i+1,
        ylim=global_ylim)
end
savefig(joinpath(@__DIR__, "../..", "plots", "SVM", problem_name, "EigenvaluesN_multiiters.pdf"))