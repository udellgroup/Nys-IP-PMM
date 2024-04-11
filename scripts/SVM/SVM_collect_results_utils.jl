using DataFrames, CSV
using Dates
using Glob

function collect_all_status!(problem_type::String, algorithm::String, 
                             problem_names::Vector{String}; saved::Bool = false)

    # Create a DataFrame to store whole results
    df = DataFrame() #create_status_df(algorithm)

    # Stem directory for results in "problem_type"
    results_dir = scriptsdir(problem_type, "results")

    # Subdirectories of results_dir are the names of the problems
    problem_names = basename.(filter(isdir, readdir(results_dir, join=true)))

    for name in problem_names
        # Define Glob matcher for status.csv files 
        matcher = Glob.GlobMatch("*_status.csv")

        # Get status files under results_dir/name/algorithm
        matched_paths = readdir(matcher, joinpath(results_dir, name, algorithm))

        # Read each status.csv files and append to df
        for result_path in matched_paths
            read_df = CSV.read(result_path, DataFrame)
            # If read_df does not have columns TotalElapsedConstructPrecond and TotalElapsedCGsolving, add them
            if !("TotalElapsedConstructPrecond" in names(read_df))
                read_df[!, "TotalElapsedConstructPrecond"] = [NaN]
            end
            if !("TotalElapsedCGsolving" in names(read_df))
                read_df[!, "TotalElapsedCGsolving"] = [NaN]
            end
            df = vcat(df, read_df)
        end
    end

    if saved
        summary_dir = scriptsdir(problem_type, "summary")
        isdir(summary_dir) || mkdir(summary_dir)  # Create directory if it does not exist

        # Save csv file with all status results
        time_stamp = Dates.format(now(), "yyyy-mm-dd--HH:MM")
        final_name = joinpath(summary_dir, time_stamp * "_All_" * algorithm * "_" * problem_type * "_status" * ".csv")
        CSV.write(final_name, df)
    end
    return df
end