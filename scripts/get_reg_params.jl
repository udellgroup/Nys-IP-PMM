using CSV
using DataFrames
using Printf

function generate_latex_table(dataname::String, csv_path::String)
    # Read the CSV file into a DataFrame
    data = CSV.read(csv_path, DataFrame)
    
    # Extract the columns "rho" and "delta"
    iter = data.iter
    rho = data.rho
    delta = data.delta
    
    # Generate the LaTeX code
    latex_table = """
    \\begin{table}[H]
    \\centering
    \\begin{tabular}{ccc}
    \\toprule
    & \\multicolumn{2}{c}{\\textbf{$(dataname)}} \\\\
    \\cmidrule(lr){2-3}
    Iteration \\(k\\) & Primal reg. \\(\\rho_k\\) & Dual reg. \\(\\delta_k\\) \\\\
    \\midrule
    """
    
    # Format each row with scientific notation
    for i in 1:length(iter)
        rho_str = @sprintf("%.2e", rho[i])    # Format rho in scientific notation
        delta_str = @sprintf("%.2e", delta[i]) # Format delta in scientific notation
        latex_table *= "    $(iter[i]) & \\num{$rho_str} & \\num{$delta_str} \\\\\n"
    end
    
    latex_table *= """
    \\bottomrule
    \\end{tabular}
    \\caption{Regularization parameters in Nys-IP-PMM.}
    \\label{tab:reg}
    \\end{table}
    """
    
    # Return the LaTeX table as a string
    return latex_table
end


function generate_latex_table_strings(dataset_names::Vector{String}, csv_paths::Vector{String})
    # Ensure the lengths of dataset_names and csv_paths match
    if length(dataset_names) != length(csv_paths)
        error("The number of dataset names and paths must match!")
    end

    # Determine the number of tables needed
    num_datasets_per_table = 5
    num_tables = ceil(Int, length(dataset_names) / num_datasets_per_table)

    table_strings = []

    for table_idx in 1:num_tables
        # Get the range of datasets for this table
        dataset_range = (num_datasets_per_table * (table_idx - 1) + 1):min(num_datasets_per_table * table_idx, length(dataset_names))
        datasets_in_table = dataset_names[dataset_range]
        paths_in_table = csv_paths[dataset_range]

        # Read the data for each dataset
        data = [CSV.read(path, DataFrame) for path in paths_in_table]

        # Determine the maximum number of iterations among the datasets
        max_iterations = maximum([nrow(d) for d in data])
        iter = 1:max_iterations

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
            latex_table *= " & \\multicolumn{2}{c}{\\textbf{$dataset}}"
        end
        latex_table *= " \\\\\n"

        # Dynamically generate \cmidrule commands
        for i in 1:num_datasets_per_table
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
csv_paths = [
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/CIFAR10/IPPMM/ts=2024-12-06--20:30:45_prob=CIFAR10_pc=Nystrom_rank=200_tol=1e-04_history.csv",
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/RNASeq/IPPMM/ts=2024-12-06--20:06:54_prob=RNASeq_pc=Nystrom_rank=200_tol=1e-04_history.csv",
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/SensIT/IPPMM/ts=2024-12-06--20:31:12_prob=SensIT_pc=Nystrom_rank=50_tol=1e-04_history.csv",
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/arcene/IPPMM/ts=2024-12-06--20:07:11_prob=arcene_pc=Nystrom_rank=20_tol=1e-04_history.csv",
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/dexter/IPPMM/ts=2024-12-06--20:06:11_prob=dexter_pc=Nystrom_rank=10_tol=1e-04_history.csv",
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/sector/IPPMM/ts=2024-12-06--20:07:09_prob=sector_pc=Nystrom_rank=20_tol=1e-04_history.csv",
    "/Users/ycchu/Documents/ResearchPapers/NysIPPMM/code/Nys-IP-PMM_publish/scripts/SVM/results/CIFAR10_1000/IPPMM/ts=2024-12-05--23:58:45_prob=CIFAR10_1000_pc=Nystrom_rank=200_tol=1e-08_history.csv"
]

dataset_names = [
    "CIFAR10",
    "RNASeq",
    # "STL10",
    "SensIT",
    "sector",
    "arcene",
    "dexter",
    "CIFAR10-1000 (section 5.2.2)"
]

tables = generate_latex_table_strings(dataset_names, csv_paths)

for (i, table) in enumerate(tables)
    println("Table $i")
    println("-"^80)
    println(table)
    println("="^80)
end