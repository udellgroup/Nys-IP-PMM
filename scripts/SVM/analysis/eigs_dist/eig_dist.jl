using DrWatson
@quickactivate "."
using LinearAlgebra
using Plots
using Measures
using LaTeXStrings
using JLD2  # Ensure you can load JLD2 files

# Include the necessary files
include(srcdir("Nys-IP-PMM.jl"))
include(scriptsdir("SVM/SVM_run_tests_utils.jl"))

# Inputs
T = Float64
problem_type = SVMProblem(T)
problem_name = "CIFAR10_1000"
total_iters = 15    # Total number of iterations (depends on the problem_name)
tol = 1e-8

# Construct A
X, y = load_data(problem_type, problem_name)
n, d = size(X)
Id = Diagonal(ones(n))
A = [Id -X * Diagonal(y); zeros(1, n) y']


# Initialization
effective_dimensions = zeros(total_iters)
eigvals_N = zeros(total_iters, n+1)

# Compute eigvals of N and effective dimension over iterations
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
plot_dir = projectdir("plots", "SVM", "eigs_dist", problem_name)
!ispath(plot_dir) ? mkpath(plot_dir) : nothing

# Plot effective dimension over iterations
plot(1:total_iters, effective_dimensions, seriestype = :line, lw = 2, marker=(:circle,5),
    xlabel = "IP-PMM iteration", ylabel = "Effective Dimension", 
    title = "Effective dimension", legend = false)
savefig(joinpath(plot_dir, "eff_dim.pdf"))

# Plot eigenvalues of AAT
plot(eigvals_AAT, seriestype = :scatter, yscale = :log10, 
            xlabel = "Index", ylabel = "Eigenvalues", 
            title = latexstring("Eigenvalues of \$AA^T\$"), 
            legend = false, markersize = 2, markerstrokewidth = 0)
savefig(joinpath(plot_dir, "EigenvaluesAAT.pdf"))


## Plot eigenvalues of AAT and those of Nₖ at 2 different iterations
# Define global y-axis limits
plot_iters = [12, 15]
global_ylim_log = log10.(extrema(vcat(eigvals_N[plot_iters, :])))
global_ylim = (10^floor(global_ylim_log[1]-1), 10^ceil(global_ylim_log[2]+1))
yticks_positions = [1, 1e5, 1e10, 1e15]  # Example positions
yticks_labels = [L"10^{0}", L"10^{5}", L"10^{10}", L"10^{15}"]  # Corresponding labels

plt = plot(layout=(1, 3), size=(1200, 400), left_margin=10mm, bottom_margin=10mm)  # 1 row, 3 columns
# Plot eigenvalues of AAT
plot!(plt, eigvals_AAT, seriestype=:scatter, yscale=:log10, 
        xlabel="Index", ylabel="Eigenvalues",
        title=L"$AA^T$", 
        legend=false, markersize=2, markerstrokewidth=0, subplot=1,
        ylim=global_ylim, yticks=(yticks_positions, yticks_labels))

# Plot eigenvalues of Nₖ at 2 different iterations
for (i, plt_iter) in enumerate(plot_iters)
    plot!(plt, eigvals_N[plt_iter, :], seriestype=:scatter, yscale=:log10, 
            xlabel="Index", 
            title=L"$N_k$ $(k = %$(plt_iter))$", 
            legend=false, markersize=2, markerstrokewidth=0, subplot=i+1,
            ylim=global_ylim, yticks=(yticks_positions, yticks_labels))
end

savefig(joinpath(plot_dir, "EigenvaluesN_multiiters.pdf"))