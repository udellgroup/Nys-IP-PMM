# Randomized Nyström Preconditioned Interior Point-Proximal Method of Multipliers (Nys-IP-PMM)
This code base uses the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Nys-IP-PMM

The project is a companion code for the article "Randomized Nyström Preconditioned Interior Point-Proximal Method of Multipliers", authored by Ya-Chi Chu, Luiz-Rafael Santos, Madeleine Udell. 
We provide the instructions to reproduce the experiments in the paper.

## Preliminary steps
0. Open a Julia console and run the following commands:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

## Large-Scale portfolio optimization experiment (Section 5.1)
1. Run the script `scripts/Portfolio/run_portfolio.jl`. It will generate the data, run the experiments for the synthetic portfolio optimization problem, and save the results in the folder `Portfolio/results/risk_model`.
2. Run the script `notebooks/Portfolio_analysis/get_demonstrate_results.jl`. It will copy the latest 6 results in `Portfolio/results/risk_model` to `notebooks/Portfolio_analysis/demonstrate_results folder` for plotting.
3. Run the notebook `notebooks/Portfolio_analysis/plotting.ipynb` to generate the plot in the paper.

## SVM experiments (Section 5.2)
0. Download the required support vector machine (SVM) datasets to a new folder `data` by running the file `scripts/SVM/SVM_data_download.jl`.

### SVM on all datasets (Section 5.2.1)
1. Run the script `scripts/SVM/SVM_run_tests.jl`. It will preprocess datasets, run the experiments, and save the results as `.csv` files under the folder `SVM/results/[dataset_name]/IPPMM`.
2. Run the script `scripts/SVM/SVM_collect_results.jl` to collect all the results in the folder `SVM/results` and save them as a summary `.csv` file under the folder `SVM/results/summary`.

### Condition numbers at different IP-PMM stages (Section 5.2.2)
0. The condition numbers are saved under `notebooks/SVM_analysis/condnum_rank`.
1. Run the section "Condition Number v.s. Rank" in the notebook `notebooks/SVM_analysis/plotting.ipynb` to generate the plot in the paper.

### Running time experiments (Section 5.2.3)
1. Run the section "Time v.s. Rank" in the notebook `notebooks/SVM_analysis/plotting.ipynb` to generate the plot in the paper.


<!-- To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are not included and need to be downloaded independently.

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
# using DrWatson
# @quickactivate "Nys-IP-PMM"
```
which auto-activate the project and enable local path handling from DrWatson. -->
