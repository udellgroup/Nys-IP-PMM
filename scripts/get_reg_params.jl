using DrWatson
@quickactivate "."
using CSV
using DataFrames
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
        paths_in_table = [scriptsdir(name2info[dataset_name]["problemtype"], "results", dataset_name, "IPPMM", name2info[dataset_name]["filename"]) for dataset_name in datasets_in_table]

        # Read the data for each dataset
        data = [CSV.read(path, DataFrame) for path in paths_in_table]

        # Determine the maximum number of iterations among the datasets
        max_iterations = maximum([nrow(d) for d in data])
        iter = 0:max_iterations-1

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
        for i in iter
            latex_table *= "    $i"
            for j in 1:length(datasets_in_table)
                if i <= nrow(data[j])  # If the dataset has data for this iteration
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
        \\caption{Regularization parameters for SVM datasets.}
        \\label{tab:reg_table_$(table_idx)}
        \\end{table}
        """
        
        # Append the generated table to the vector of strings
        push!(table_strings, latex_table)
    end

    return table_strings
end

##
name2info = Dict(
    "CIFAR10" => Dict(
        "filename" => "ts=2024-12-07--01:26:18_prob=CIFAR10_pc=Nystrom_rank=200_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{CIFAR10}"
    ),
    "RNASeq" => Dict(
        "filename" => "ts=2024-12-07--00:33:49_prob=RNASeq_pc=Nystrom_rank=200_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{RNASeq}"
    ),
    "STL10" => Dict(
        "filename" => "ts=2024-12-07--09:48:04_prob=STL10_pc=Nystrom_rank=800_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{STL10}"
    ),
    "SensIT" => Dict(
        "filename" => "ts=2024-12-07--01:26:59_prob=SensIT_pc=Nystrom_rank=50_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{SensIT}"
    ),
    "sector" => Dict(
        "filename" => "ts=2024-12-07--00:34:19_prob=sector_pc=Nystrom_rank=20_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{sector}"
    ),
    "arcene" => Dict(
        "filename" => "ts=2024-12-07--00:34:22_prob=arcene_pc=Nystrom_rank=20_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{arcene}"
    ),
    "dexter" => Dict(
        "filename" => "ts=2024-12-07--00:32:22_prob=dexter_pc=Nystrom_rank=10_tol=1e-04_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\textbf{dexter}"
    ),
    "CIFAR10_1000" => Dict(
        "filename" => "ts=2024-12-06--23:24:18_prob=CIFAR10_1000_pc=Nystrom_rank=200_tol=1e-08_history.csv",
        "problemtype" => "SVM",
        "tablename" => "\\begin{tabular}[c]{@{}c@{}}\\textbf{CIFAR10\\_1000} \\\\ \\textbf{(section 5.2.2)}\\end{tabular}"
    ),
    "risk_model" => Dict(
        "filename" => "ts=2024-12-08--11:45:02_prob=risk_model_m=500_n=800_k=10_pc=Nystrom_rank=20_tol=1e-08_history.csv",
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