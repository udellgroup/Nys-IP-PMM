using DrWatson
@quickactivate "."
using CSV
using DataFrames
using Glob
using Printf

function generate_latex_table_strings(dataset_names::Vector{String}, name2info::Dict{String, Dict{String, String}})
    # Determine the number of tables needed
    n_datasets = length(dataset_names)
    num_datasets_per_table = 5
    num_tables = ceil(Int, n_datasets / num_datasets_per_table)

    table_strings = []
    for table_idx in 1:num_tables
        # Get the range of datasets for this table
        dataset_range = (num_datasets_per_table * (table_idx - 1) + 1):min(num_datasets_per_table * table_idx, n_datasets)
        datasets_in_table = dataset_names[dataset_range]
        paths_in_table = [ 
            joinpath(
                get_results_csv_dir(name2info[dataset_name]["problemtype"], dataset_name), 
                get_latest_history_csv_filename(name2info[dataset_name]["problemtype"], dataset_name)
            ) for dataset_name in datasets_in_table
        ]

        # Read the data for each dataset
        data = [CSV.read(path, DataFrame) for path in paths_in_table]

        # Determine the maximum number of iterations among the datasets
        max_rows = maximum([nrow(d) for d in data])
        rows = 1:max_rows

        # Generate the LaTeX table header
        latex_table = """
        \\begin{table}[H]
        \\centering
        \\resizebox{\\linewidth}{!}{
        \\begingroup
        \\begin{tabular}{c""" * "cc"^length(datasets_in_table) * """}
        \\toprule
        """

        # Add the dataset names as supercolumns
        for dataset in datasets_in_table
            latex_table *= " & \\multicolumn{2}{c}{$(name2info[dataset]["tablename"])}"
        end
        latex_table *= " \\\\\n"

        # Dynamically generate \cmidrule commands
        for i in 1:length(paths_in_table)
            start_col = 2 * i  # Start column for this dataset
            end_col = start_col + 1  # End column for this dataset
            latex_table *= "\\cmidrule(lr){" * string(start_col) * "-" * string(end_col) * "} "
        end
        latex_table *= "\n"

        # latex_table *= "\\cmidrule(lr){2-3}" * " \\cmidrule(lr){4-5}" * " \\cmidrule(lr){6-7}" * "\n"

        # Add the subcolumn headers
        latex_table *= "\\(k\\)"
        for _ in datasets_in_table
            latex_table *= " & \\(\\rho_k\\) & \\(\\delta_k\\)"
        end
        latex_table *= " \\\\\n\\midrule\n"

        # Add the data rows
        for i in rows
            latex_table *= "    $(i-1)"
            for j in 1:length(datasets_in_table)
                curr_rows = nrow(data[j])
                if i <= curr_rows  # If the dataset has data for this iteration
                    rho_str = @sprintf("%.2e", data[j].rho[i])
                    delta_str = @sprintf("%.2e", data[j].delta[i])
                    latex_table *= " & \\num{$rho_str} & \\num{$delta_str}"
                else  # Leave the entry blank if the dataset doesn't have data for this iteration
                    latex_table *= " & -- & -- "
                end
            end
            latex_table *= " \\\\\n"
        end

        # Close the LaTeX table
        latex_table *= """
        \\bottomrule
        \\end{tabular}
        \\endgroup
        }
        \\caption{Regularization parameters for for experiments in \\cref{sec:numerical-exp}.}
        \\label{tab:reg_table_$(table_idx)}
        \\end{table}
        """
        
        # Append the generated table to the vector of strings
        push!(table_strings, latex_table)
    end

    return table_strings
end

get_results_csv_dir(problem_type::String, problem_name::String) = scriptsdir(problem_type, "results", problem_name, "IPPMM")

function get_latest_history_csv_filename(problem_type::String, problem_name::String)
    # Get the directory containing the results
    results_dir = get_results_csv_dir(problem_type, problem_name)
    
    # Define Glob matcher for history.csv files
    matcher = Glob.GlobMatch("*" * problem_name * "*Nystrom*_history.csv")

    # Get all history files under results_dir/name/algorithm
    matched_paths = readdir(matcher, results_dir)
    @assert !isempty(matched_paths) "No history files found for $problem_name"

    # Find the latest file
    latest_idx = argmax(mtime.(matched_paths))
    latest_file = matched_paths[latest_idx]

    return basename(latest_file)
end
##
name2info = Dict(
    "CIFAR10" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{CIFAR10}"
    ),
    "RNASeq" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{RNASeq}"
    ),
    "STL10" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{STL10}"
    ),
    "SensIT" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{SensIT}"
    ),
    "sector" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{sector}"
    ),
    "arcene" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{arcene}"
    ),
    "dexter" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\textbf{dexter}"
    ),
    "CIFAR10_1000" => Dict(
        "problemtype" => "SVM",
        "tablename" => "\\begin{tabular}[c]{@{}c@{}}\\textbf{CIFAR10\\_1000} \\\\ \\textbf{(section 5.2.2)}\\end{tabular}"
    ),
    "risk_model" => Dict(
        "problemtype" => "Portfolio",
        "tablename" => "\\begin{tabular}[c]{@{}c@{}}\\textbf{Portfolio} \\\\ \\textbf{(section 5.1)}\\end{tabular}"
    ),
)

dataset_names = [
    "risk_model",
    "CIFAR10_1000",
    "RNASeq", 
    "SensIT", 
    "sector", 
    "CIFAR10", 
    "STL10", 
    "arcene", 
    "dexter", 
]

tables = generate_latex_table_strings(dataset_names, name2info)

for (i, table) in enumerate(tables)
    println("Table $i")
    println("-"^80)
    println(table)
    println("="^80)
end