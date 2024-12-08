using CSV
using DataFrames
using Dates

abstract type AbstractIPMProblem end
abstract type AbstractMPSProblem <: AbstractIPMProblem end

"""
    test_IPPMM()

Run test function for IPPMM.
"""
function test_IPPMM(problem_type::AbstractIPMProblem, 
                    problem_name::String, 
                    method_P_list::Vector, tol=1e-4; 
                    krylov_tol = 1e-6, maxit::Int = 25, timed::Bool = true, saved::Bool = true)
    println("-"^110)
    
    # Do not time if saved is false
    timed = saved ? timed : false

    # Get IPMInput and IPPMMParams
    input, params = get_IPMInput_IPPMMParams(problem_type, problem_name, tol)

    # Update IPPMMParams with krylov_tol and adaptive_sketch
    params.krylov_tol = krylov_tol

    # Allocate memory for CG solver
    Vx = typeof(input.c)
    nrow, ncol = size(input.opA)
    cg_solver = CgSolver(nrow, nrow, Vx)    

    # Reproducibility
    Random.seed!(1234)

    # Construct an initial point
    T = eltype(input.c)
    @printf("Obtaining initial point... ")
    initpt_info = IPPMM_InitStatus()   # Initialize cgiter = 0, time = 0.0
    initial_point = IPMVariables{T}(nrow, ncol)
    initpt_info.cgiter = obtain_initial_point!(initial_point, input; cg_solver=cg_solver)
    println("Successfully obtained initial point.")

    # Get time for obtaining initial point
    if timed 
        println("Computing initial point again to get time...")
        initpt_info.time = @elapsed obtain_initial_point!(initial_point, input; cg_solver=cg_solver, printlevel=0)
        println("Successfully computed initial point again.\n", "="^110)
    else 
        println("Not re-compute initial point for running time.\n", "="^110)
    end

    # Run IP-PMM
    vars = nothing
    for method_P in method_P_list

        # Reset rank for Nystrom and PartialCholesky if not specified
        if ((typeof(method_P) <: method_Nystrom && method_P.sketchsize == 0) || 
            (typeof(method_P) <: method_PartialCholesky && method_P.k_steps == 0))
            reset_rank!(method_P, nrow)
        end

        # Record the start sketchsize for Nystrom
        start_sketchsize = (typeof(method_P) <: method_Nystrom) ? method_P.sketchsize : 0
        # Initialize A for PartialCholesky
        A = nothing
                
        # First run to get status
        print_running_info(problem_name, method_P)
        time_IPPMM = @elapsed begin
            # Construct A matrix for PartialCholesky
            if (typeof(method_P) <: method_PartialCholesky)
                println("Construct constraint matrix A (for Partial Cholesky preconditioner) of ", problem_name, "...")
                A = get_A_matrix(problem_type, problem_name)
                println("Successfully constructed A for ", problem_name, ".")
            end
            
            # Run IP-PMM
            history, opt, vars = IP_PMM_bdd(input; initial_point=initial_point, params=params,
                                            method_P=method_P, 
                                            tol=tol, maxit = maxit, pc=true, printlevel = 3, A = A);
        end
        println("First run takes ", time_IPPMM, " seconds.\n")

        # Run IP-PMM again to get time
        if (opt == :Solved) && timed
            println("-"^110, "\n", "Running again to get time...\n")
            
            # Reset sketchsize to initial sketchsize for Nystrom
            if (typeof(method_P) <: method_Nystrom) 
                method_P.sketchsize = start_sketchsize
            end

            # Second run to time
            time_IPPMM = @elapsed begin
                # Construct A matrix for PartialCholesky
                if (typeof(method_P) <: method_PartialCholesky)
                    A = get_A_matrix(problem_type, problem_name)
                end
                
                # Run IP-PMM
                history, opt, vars = IP_PMM_bdd(input; initial_point=initial_point, params=params,
                                                method_P=method_P, 
                                                tol=tol, maxit = maxit, pc=true, printlevel = 0, A = A);
            end
        end
        println("Size of iter: ", length(history["iter"]))
        println("Size of CG Elapsed: ", length(history["CG_solving_elapsed"]))

        # Save results to csv file
        if saved
            print_saving_info(problem_name, method_P)
            preconditioner, rank = unwrap_method_P(method_P)
            IPPMM_args = IPPMMargs(preconditioner, rank, tol, maxit)
            IPPMM_summary = IPPMMSummary(history, opt, time_IPPMM)
            save_IPPMM_csv(problem_type, problem_name, nrow, IPPMM_args, initpt_info, IPPMM_summary)
            println("Successfully saved results.\n", "="^110)
        end
    end
    return vars
end

function print_running_info(problem_name, method_P)
    type_method_P = typeof(method_P)
    if (type_method_P <: method_Nystrom)
        println("Running $problem_name with Nystrom preconditioner and sketchsize $(method_P.sketchsize) ...")
    elseif (type_method_P <: method_PartialCholesky)
        println("Running $problem_name with Partial Cholesky preconditioner and target rank $(method_P.k_steps) ...")
    elseif (type_method_P <: method_NoPreconditioner)
        println("Running $problem_name with No preconditioner ...")
    end
end

function print_saving_info(problem_name, method_P)
    type_method_P = typeof(method_P)
    if (type_method_P <: method_Nystrom)
        println("Saving results for $problem_name with Nystrom preconditioner and sketchsize $(method_P.sketchsize) ...")
    elseif (type_method_P <: method_PartialCholesky)
        println("Saving results for $problem_name with Partial Cholesky preconditioner and target rank $(method_P.k_steps) ...")
    elseif (type_method_P <: method_NoPreconditioner)
        println("Saving results for $problem_name with No preconditioner ...")
    end
end

function unwrap_method_P(method_P)
    type_method_P = typeof(method_P)
    if (type_method_P <: method_Nystrom)
        return :Nystrom, method_P.sketchsize
    elseif (type_method_P <: method_PartialCholesky)
        return :PartialCholesky, method_P.k_steps
    elseif (type_method_P <: method_NoPreconditioner)
        return :NoPreconditioner, 0
    end
end

function reset_rank!(method_P, nrow)
    rank = nothing
    if nrow > 5000
        rank = 100
    else
        rank = max(floor(Int, nrow/50), 10)
        if rank > nrow
            rank = floor(Int, nrow/2)
        end
    end
    
    if (typeof(method_P) <: method_Nystrom)
        method_P.sketchsize = rank
    elseif (typeof(method_P) <: method_PartialCholesky)
        method_P.k_steps = rank
    end
    return nothing
end


## 
"""
    mutable struct IPPMMargs

Arguments for IPPMM.
"""
Base.@kwdef struct IPPMMargs
    preconditioner::Symbol
    rank::Int
    tol::Float64
    maxiter::Int
end


"""
    mutable struct IPPMM_InitStatus

Information for solving the initial point of IPPMM.
args:
    cgiter: number of CG iterations
    time: time for solving the initial point
"""
Base.@kwdef mutable struct IPPMM_InitStatus
    cgiter::Int
    time::Float64
end
IPPMM_InitStatus() = IPPMM_InitStatus(0, 0.0)


"""
    mutable struct IPPMMSummary

Output summary of IPPMM.
args:
    history: history of IPPMM including primal/dual feasibility, optimality gap, etc.
    opt: status of IPPMM (solved, unsolved, etc.)
    time: time for solving the problem
"""
Base.@kwdef struct IPPMMSummary
    history::OrderedDict
    opt::Symbol
    time::Float64
end


"""
    mutable struct IPPMM_CSVInfo

Information for saving the results of IPPMM.
"""
mutable struct IPPMM_CSVInfo
    problem_class::String
    problem_name::String
    normal_eq_size::Int
    IPPMM_args::IPPMMargs
    initpt_info::IPPMM_InitStatus
    IPPMM_summary::IPPMMSummary
end



##
function create_status_df(time_stamp, problem_name, IPPMM_args, IPPMM_summary, initpt_info, nrow)
    # Compute total inner iterations
    total_inner_iter = sum(IPPMM_summary.history["inner_iter_predictor"]) + sum(IPPMM_summary.history["inner_iter_corrector"])   
    total_construct_precond_elapsed = sum(IPPMM_summary.history["construct_precond_elapsed"])
    total_CG_elpased = sum(IPPMM_summary.history["CG_solving_elapsed"])

    return DataFrame("TimeStamp"            => time_stamp,
                     "Name"                 => problem_name,
                     "Preconditioner"       => IPPMM_args.preconditioner,
                     "rank"                 => IPPMM_args.rank,
                     "tol"                  => IPPMM_args.tol,
                     "maxiter"              => IPPMM_args.maxiter,
                     "Status"               => IPPMM_summary.opt,
                     "IPPMM_iter"           => IPPMM_summary.history["iter"][end], 
                     "IPPMM_time"           => IPPMM_summary.time,
                     "sum_inner_iter"       => total_inner_iter,
                     "final_primal_feas"    => IPPMM_summary.history["primal_feasibility"][end],
                     "final_dual_feas"      => IPPMM_summary.history["dual_feasibility"][end],
                     "optimality_gap"       => IPPMM_summary.history["optimality_gap"][end],
                     "StartPt_iter"         => initpt_info.cgiter, 
                     "StartPt_time"         => initpt_info.time, 
                     "NormalEq_size"        => nrow,
                     "TotalElapsedConstructPrecond" => total_construct_precond_elapsed,
                     "TotalElapsedCGsolving" => total_CG_elpased)
end

function save_IPPMM_csv(problem_type, problem_name::String, nrow, IPPMM_args, initpt_info, IPPMM_summary)
    # Create (if not existed) saving directory: scripts/{problem_type}/results/{problem_name}/IPPMM
    if typeof(problem_type) <: AbstractMPSProblem
        if problem_type.presolved
            destination = scriptsdir(get_class_name(problem_type), "results", "Presolved", problem_name, "IPPMM")
        else
            destination = scriptsdir(get_class_name(problem_type), "results", "NonPresolved", problem_name, "IPPMM")
        end
    else
        destination = scriptsdir(get_class_name(problem_type), "results", problem_name, "IPPMM")
    end
    isdir(destination) ? nothing : mkpath(destination)
    
    # Get time stamp
    time_stamp = Dates.format(now(), "yyyy-mm-dd--HH:MM:SS")

    # Get names for two csv files: history.csv and status.csv
    history_filename, status_filename = get_csvnames(time_stamp, problem_type, problem_name, IPPMM_args)
    
    # Write history.csv
    CSV.write(joinpath(destination, history_filename), IPPMM_summary.history)
    
    # Write status.csv
    status_df = create_status_df(time_stamp, problem_name, IPPMM_args, IPPMM_summary, initpt_info, nrow)
    CSV.write(joinpath(destination, status_filename), status_df)
    
    return nothing
end







## Printing functions
function print_header(pl, pc, method_P)
    type_method_P = typeof(method_P)
    if (pl >= 1)
        if (pc == true)
            @printf("%49s  ", "inner iter")
            @printf("%11s  ", "inner iter")
            @printf("\n")
        end
        @printf(" ")
        @printf("%4s    ", "iter")
        @printf("%8s  ", "pr feas")
        @printf("%8s  ", "dl feas")
        @printf("%8s  ", "μ")
        if (pc == true)
            @printf("%11s  ", "(predictor)")
            @printf("%11s  ", "(corrector)")
        else
            @printf("%11s  ", "inner iter")
        end
    end
    if (pl >= 2)
        @printf("  ")
        @printf("%10s  ", "krylov_tol")
        if (type_method_P <: method_Nystrom)
            @printf("%10s  ", "sketchsize")
        elseif (type_method_P <: method_PartialCholesky)
            @printf("%10s  ", "targetrank")
        end
    end
    if (pl >= 3)
        @printf("  ")
        @printf("%8s  ", "α_primal")
        @printf("%8s  ", "α_dual")
        @printf("%8s  ", "ρ")
        @printf("%8s  ", "δ")
    end
    if (pl >= 1)
        if (pc == true)
            @printf("\n ====    ========  ========  ========  ===========  ===========")
        else
            @printf("\n ====    ========  ========  ========  ===========")
        end
    end
    if (pl >= 2)
        @printf("    ==========")
        (type_method_P <: method_NoPreconditioner) ? nothing : @printf("  ==========")
    end
    if (pl >= 3)
        @printf("    ========  ========  ========  ========")
    end
    if (pl >= 1)
        @printf("\n")
    end
end

function print_output(pl, pc, method_P, it, xinf, sinf, μ, inneriter, krylov_tol, α_primal, α_dual, ρ, δ)
    type_method_P = typeof(method_P)
    if (type_method_P <: method_Nystrom)
        rank = method_P.sketchsize
    elseif (type_method_P <: method_PartialCholesky)
        rank = method_P.k_steps
    end

    if (pl >= 1)
        @printf(" ")
        @printf("%4d    ", it)
        @printf("%8.2e  ", xinf)
        @printf("%8.2e  ", sinf)
        @printf("%8.2e  ", μ)
        if it == 0
            @printf("%11s  ", "")
        elseif (pc == true)
            @printf("%11d  ", inneriter[1])
            @printf("%11d  ", inneriter[2])
        else
            @printf("%11d  ", inneriter)
        end
    end
    if (pl >= 2)
        @printf("  ")
        if it == 0
            @printf("%10s  ", "")
            (type_method_P <: method_NoPreconditioner) ? nothing : @printf("%10s  ", "")
        else
            @printf("%10.2e  ", krylov_tol)
            (type_method_P <: method_NoPreconditioner) ? nothing : @printf("%10d  ", rank)
        end
    end
    if (pl >= 3)
        @printf("  ")
        if it == 0
            @printf("%8s  ", "")
            @printf("%8s  ", "")
        else
            @printf("%8.2e  ", α_primal)
            @printf("%8.2e  ", α_dual)
        end
        @printf("%8.2e  ", ρ)
        @printf("%8.2e  ", δ)
    end
    if (pl >= 1)
        @printf("\n")
    end
end
