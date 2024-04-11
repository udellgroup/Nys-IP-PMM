using DrWatson
@quickactivate "."

include(scriptsdir("SVM", "SVM_collect_results_utils.jl"))

problem_type = "SVM"
algorithm = "IPPMM"

results_dir = scriptsdir(problem_type, "results")
problem_names = basename.(filter(isdir, readdir(results_dir, join=true)))

df = collect_all_status!(problem_type, algorithm, problem_names, saved = true)