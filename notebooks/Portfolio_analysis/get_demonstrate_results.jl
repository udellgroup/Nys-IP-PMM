####################################################################################################
## Copy the latest 6 results to notebooks/Portfolio_analysis/demonstrate_results folder
####################################################################################################
using DrWatson
@quickactivate "."

# Define the source and destination paths
source_path = scriptsdir("Portfolio", "results", "risk_model", "IPPMM")
destination_path = projectdir("notebooks", "Portfolio_analysis", "demonstrate_results")

# Create the destination folder if it does not exist
if !isdir(destination_path)
    mkpath(destination_path)
end

# Get a list of all CSV files in the source path
csv_files = filter(x -> x[end-3:end] == ".csv", readdir(source_path))

# Get the latest 6 files in the source path
latest_files = sort(collect(readdir(source_path)), rev=true)[1:6]

# Copy each of the latest 6 CSV files to the destination path
for file in latest_files
    source_file = joinpath(source_path, file)
    destination_file = joinpath(destination_path, file)
    cp(source_file, destination_file)
end