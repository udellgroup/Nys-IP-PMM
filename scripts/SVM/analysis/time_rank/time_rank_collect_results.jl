using DrWatson
@quickactivate "."
using JLD2
using CSV, Glob
using Dates
using DataFrames
using Statistics


problem_name = "RNASeq"
n_rounds = 5

# Collect all the status results from different rounds
dfs = []
for round in 1:n_rounds
    # Define the directory for the round
    round_results_dir = scriptsdir("SVM", "analysis", "time_rank", "raw_results", problem_name, "Round$(round)")
    
    # Read all the csv files in the directory
    matcher = Glob.GlobMatch("*_status.csv")
    files = readdir(matcher, round_results_dir)
    for file in files
        if occursin(".csv", file)
            result_path = joinpath(round_results_dir, file)
            push!(dfs, CSV.read(result_path, DataFrame))
        end
    end
end
# Concatenate all DataFrames
round_df = vcat(dfs...)

# Group the dataframe by "Preconditioner", and "rank" and calculate the average of "IPPMM_time", "TotalElapsedConstructPrecond", and "TotalElapsedCGsolving"
averaged_df = combine(groupby(round_df, [:Preconditioner, :rank]), 
                        :IPPMM_time => mean => :IPPMM_time,
                        :TotalElapsedConstructPrecond => mean => :TotalElapsedConstructPrecond,
                        :TotalElapsedCGsolving => mean => :TotalElapsedCGsolving)


# Save the averaged results to csv files in two directories
filedirs = [
    scriptsdir("SVM", "analysis", "time_rank", "averaged_results"),
    projectdir("notebooks", "SVM_analysis", "time_rank"),
]
time_stamp = Dates.format(now(), "yyyy-mm-dd--HH:MM:SS")
filename_averaged = time_stamp * "_" * problem_name * "_averaged_time.csv"
for dir in filedirs
    !ispath(dir) ? mkpath(dir) : nothing
    CSV.write(joinpath(dir, filename_averaged), averaged_df)
end