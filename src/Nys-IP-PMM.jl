using DrWatson
@quickactivate "Nys-IP-PMM"
if occursin("Intel", Sys.cpu_info()[1].model)
    using MKL
end
using DataStructures
using LinearAlgebra, SparseArrays, Krylov, LinearOperators
using Printf
import Base.promote_eltypeof
import LinearOperators.storage_type
import LinearAlgebra: cond
const LO = LinearOperators

include(srcdir("IP-PMM_structs.jl"))
include(srcdir("normal_equations.jl"))
include(srcdir("IP-PMM_helper-func.jl"))
include(srcdir("run_test_utils.jl"))
include(srcdir("preconditioners.jl"))
##


function IP_PMM_bdd(input::IPMInput{T};
                    initial_point::IPMVariables{T} = IPMVariables{T}(0,0),
                    params::IPPMMParams = IPPMMParams(T),
                    method_P = method_NoPreconditioner{T}(),
                    tol::T = 1e-6, maxit::Int64 = 30,
                    pc::Bool = false, printlevel::Int64 = 1,
                    A::Union{AbstractMatrix{T}, Nothing} = nothing) where T <: Number
    cumulative_time = 0.0
    cumulative_time += @elapsed begin
    # Create dict for storing results
    history = OrderedDict(
        "iter"                    => Int[], 
        "primal_feasibility"      => Float64[], 
        "dual_feasibility"        => Float64[], 
        "optimality_gap"          => Float64[],
        "inner_iter_predictor"    => Int[], 
        "inner_iter_corrector"    => Int[],
        "krylov_tol"              => Float64[], 
        "rank"                    => Int[],
        "construct_precond_elapsed" => Float64[],
        "CG_solving_elapsed"      => Float64[],
        "cumulative_time"         => Float64[],
        "rho"                     => Float64[],
        "delta"                   => Float64[]
    )
    # Unwrap input
    m, n = input.nrow, input.ncol
    opA = @views input.opA
    b = @views input.b
    c = @views input.c
    indices = @views input.indices
    u = @views input.u
    opQ = @views input.opQ
    diagQ = @views input.diagQ

    # Unwrap params
    normal_eq = @views params.normal_eq
    krylov_tol = @views params.krylov_tol
    max_sketchsize = @views params.max_sketchsize
    reg_limit = @views params.reg_limit

    tol_reg_limit = 5e-13

    # Always use normal equation if solving LP
    (input.prob_type == :LP) ? (normal_eq = true) : nothing
    
    # Check if it is normal equation. Otherwise, throw an error.
    if !normal_eq
        @error("Only can solve IP-PMM with normal equations.")
    end

    normal_eq && @assert !isempty(diagQ) "For normal equation, Q should be diagonal and diagQ must be provided."

    # Check if the input is valid
    dim_checking(m, n, c, opQ, b, u, indices)
    
    n_IJ = length(indices.normal) + length(indices.box)  # number of all positive variables (= |I| + |J|)

    type_method_P = typeof(method_P)

    opAT = opA'
    pl = printlevel     
    ((n_IJ == 0) && (pc ≠ false)) ? pc = false : nothing    # Turn off Predictor-Corrector when PMM is only running.
    
    vars = IPMVariables{T}(m, n)
    x = @views vars.x
    y = @views vars.y
    z = @views vars.z
    w = @views vars.w
    s = @views vars.s

    steps = IPMVariables{T}(m, n)
    dx = @views steps.x
    dy = @views steps.y
    dz = @views steps.z
    dw = @views steps.w
    ds = @views steps.s

    if pc ≠ false
        steps_c = IPMVariables{T}(m, n)
        dx_c = @views steps_c.x
        dy_c = @views steps_c.y
        dz_c = @views steps_c.z
        dw_c = @views steps_c.w
        ds_c = @views steps_c.s
    end

    # Time per iteration for the construction of the preconditioner and the CG solver.
    global construct_precond_elapsed = zero(T)
    global CG_solving_elapsed = zero(T)

    # Initial Point
    Vx = typeof(c) 
    if isempty(initial_point)
        cg_solver = CgSolver(m, m, Vx)      # Allocate Memory for the CG solver.
        opN_Reg = opRegNormalEquations(opA, δ) # Allocate Memory for the Normal Equations operator.
        obtain_initial_point!(vars, input; sketchsize = sketchsize, krylov_tol = krylov_tol,
                              cg_solver = cg_solver, printlevel = printlevel)
    else
        copyto!(vars, initial_point)
    end
    
    # ==================================================================================================================== #  
    # Initialize parameters
    # -------------------------------------------------------------------------------------------------------------------- #
    iter = 0;   opt = :Unsolved;
    α_primal, α_dual = zero(T), zero(T)     # Step-length for primal/dual variables (initialization)
    σmin, σmax = 0.05*one(T), 0.95*one(T)   # Heuristic values.
    σ = zero(T)
    
    res = IPMResiduals{T}(m, indices)
    res_p = @views res.primal
    res_d = @views res.dual
    res_u = @views res.upper
    res_xz = @views res.compl_xz
    res_ws = @views res.compl_ws


    # Initialize μ and res_μ based on the presence of non-negativity constraints.
    (n_IJ > 0) ? (μ = compute_μ(vars, indices)) : (μ = zero(T))     # Initial value of μ.
    # (n_IJ > 0) ? (res_μ = zeros(T, n)) : (res_μ = Vector{T}(undef, 0, 0))

    print_header(pl, pc, method_P)    # Set the printing choice.

    if (pc == false)
        inneriter = 0
        retry = 0       # Num of times a factorization is re-built (for different regularization values)
    else
        inneriter = [0, 0]
        inneriter_jump = [0, 0]
        adaptive_info = Dict("inneriter" => inneriter, "inneriter_jump" => inneriter_jump)
        sum_inner_iter = [0, 0]
        retry_p = 0
        retry_c = 0
    end
    max_tries = 10      # Maximum number of times before exiting with an ill-conditioning message.
    μ_prev = zero(T)
    
    δ, ρ = 8*one(T), 8*one(T)
    λ, ζ = copy(y), copy(x)             # Initial estimates of the Lagrange multipliers AND primal optimal solution
    no_dual_update = 0      # Primal infeasibility detection counter.
    no_primal_update = 0    # Dual infeasibility detection counter.
    new_nr_res_p = zeros(T, m)
    new_nr_res_d = zeros(T, n)
    nr_res_p = zeros(T, m)
    nr_res_d = zeros(T, n)

    # Allocate memory for Krylov solver
    if !(@isdefined cg_solver)
        construct_precond_elapsed += @elapsed begin
            cg_solver = CgSolver(m, m, Vx) # Allocate Memory for the CG solver.
            opN_Reg = opRegNormalEquations(opA, δ) # Allocate Memory for the Normal Equations operator.
        end
    end

    # Allocate memory for preconditioner if necessary
    construct_precond_elapsed += @elapsed Pinv = allocate_preconditioner(method_P, opN_Reg)

    end
    # ==================================================================================================================== #

    while iter < maxit

        cumulative_time += @elapsed begin
            if iter > 1
                copyto!(nr_res_p, new_nr_res_p)
                copyto!(nr_res_d, new_nr_res_d)
            else
                # nr_res_p1 = b - opA*x                                # Non-regularized primal residual
                mul!(nr_res_p, opA, x, -one(T), zero(T))
                @. nr_res_p += b                             # Non-regularized dual residual
                # nr_res_d = opQ * x - opAT * y - z + s + c 
                mul!(nr_res_d, opQ, x)
                mul!(nr_res_d, opAT, y, -one(T), one(T))
                @. nr_res_d -= z
                @. nr_res_d += s
                @. nr_res_d += c
                                    # Non-regularized dual residual.
                if any(isnan.(nr_res_p)) || any(isnan.(nr_res_d))
                    @error("NaN detected in the initial residual.")
                    opt = :error
                    break;
                end
            end
            # ================================================================================================================#
            # Print initial iteration (iter = 0) output AND push to history
            # ---------------------------------------------------------------------------------------------------------------- #
            if iter == 0
                pres_inf = norm(nr_res_p)/max(100,norm(b))
                dres_inf = norm(nr_res_d)/max(100,norm(c))
                μ = compute_μ(vars, indices)
                print_output(pl, pc, method_P, iter , pres_inf, dres_inf, μ, nothing, nothing, nothing, nothing, ρ, δ);
                
                # Update the history dictionary
                push!(history["iter"], iter)
                push!(history["primal_feasibility"], pres_inf)
                push!(history["dual_feasibility"], dres_inf)
                push!(history["optimality_gap"], μ)
                if pc == true
                    push!(history["inner_iter_predictor"], 0)
                    push!(history["inner_iter_corrector"], 0)
                else
                    push!(history["inner_iter_predictor"], 0)
                    push!(history["inner_iter_corrector"], 0)
                end
                push!(history["krylov_tol"], 0.0)
                push!(history["rho"], ρ)
                push!(history["delta"], δ)
    
                if (type_method_P <: method_Nystrom)
                    push!(history["rank"], 0)
                elseif (type_method_P <: method_PartialCholesky)
                    push!(history["rank"], 0)
                elseif (type_method_P <: method_NoPreconditioner)
                    push!(history["rank"], 0)
                end
                push!(history["construct_precond_elapsed"], 0.0)
                push!(history["CG_solving_elapsed"], 0.0)
                push!(history["cumulative_time"], 0.0)
            end

            # ===============================================================================================================
            
            y_minus_λ = y - λ
            x_minus_ζ = x - ζ
            @. res_p = nr_res_p - δ * (y_minus_λ)         # Regularized primal residual.
            @. res_d = nr_res_d + ρ * (x_minus_ζ)         # Regularized dual residual.
            @views @. res_u = u - x[indices.box] - w[indices.box]  # Upper bound residual.
            # ================================================================================================================#
            # Check termination criteria
            # ---------------------------------------------------------------------------------------------------------------- #
            if (norm(nr_res_p)/max(100,norm(b)) < tol) && (norm(nr_res_d)/max(100,norm(c)) < tol) &&  (μ < tol)
                pl >= 1 && @printf("optimal solution found\n")
                opt = :Solved
                break;
            end
            if (norm(y_minus_λ) > 1e10) && (norm(res_p) < tol) && (no_dual_update > 5)
                pl >= 1 && @printf("The primal-dual problem is infeasible\n")
                opt = :Infeasible_1
                break;
            end
            if (norm(x_minus_ζ) > 1e10) && (norm(res_d) < tol) && (no_primal_update > 5)
                pl >= 1 &&  @printf("The primal-dual problem is infeasible\n")
                opt = :Infeasible_2
                break;
            end
            # ================================================================================================================#
            iter += 1
            # ================================================================================================================
            
            if (no_primal_update > 5) && (ρ == reg_limit) && (reg_limit != tol_reg_limit)
                reg_limit = tol_reg_limit
                no_primal_update = 0
                no_dual_update = 0
            elseif (no_dual_update > 5) && (δ == reg_limit) && (reg_limit != tol_reg_limit)
                reg_limit = tol_reg_limit
                no_primal_update = 0
                no_dual_update = 0
            end
            # ================================================================================================================#
            # ================================================================================================================#
            # Find the preconditioner.
            # ---------------------------------------------------------------------------------------------------------------- #
            construct_precond_elapsed += @elapsed begin  
                update_opN_Reg!(opN_Reg, diagQ, ρ, δ,  vars, indices)
                update_preconditioner!(method_P, Pinv, opN_Reg, A, adaptive_info)
            end

            # Save opN_Reg.opN.D.diag as a JLD2 file
            filepath = scriptsdir("SVM", "results", "CIFAR10_1000", "IPPMM", "diagD", "CIFAR10_1000_tol=$(tol)_iter=$(iter)_diagD.jld2")
            save(filepath, Dict("diagD" => opN_Reg.opN.D.diag, "delta" => δ))
            # ================================================================================================================#
            # Mehrotra predictor-corrector.
            # ================================================================================================================#
            # Predictor step: Set sigma = 0. Solve the Newton system and compute a centrality measure.
            # ---------------------------------------------------------------------------------------------------------------- #
            @views @. res_xz[indices.normal] = - x[indices.normal] * z[indices.normal]
            @views @. res_xz[indices.box] = - x[indices.box] * z[indices.box]
            @views @. res_ws[indices.box] = - w[indices.box] * s[indices.box]
            # ============================================================================================================ #
            # Solve the Newton system with the predictor right hand side -> Optimistic view, solve as if you wanted to 
            #                                                               solve the original problem in 1 iteration.
            # ------------------------------------------------------------------------------------------------------------ #
            CG_solving_elapsed += @elapsed begin 
                instability, inneriter_pred = normal_eq_solve!(steps, cg_solver, opN_Reg, Pinv, vars, res, indices, krylov_tol)
            end

            if instability == true        # Checking if the matrix is too ill-conditioned. Mitigate it.
                if retry_p < max_tries
                    @printf("The system is re-solved, due to bad conditioning of predictor system.\n")
                    δ = δ * 100
                    ρ = ρ * 100
                    iter -= 1
                    retry_p += 1
                    reg_limit = min(reg_limit*10, tol)
                    continue;
                else
                    @printf("The system matrix is too ill-conditioned.\n");
                    break;
                end
            end
            retry_p = 0
            # ============================================================================================================ #
            
            # ============================================================================================================ #
            # Step in the non-negativity orthant.
            # ------------------------------------------------------------------------------------------------------------ #
            α_primal, α_dual = stepsize_in_orthant(vars, steps, indices)
            # ============================================================================================================ #
            μ_aff = compute_μ(x + α_primal .* dx, z + α_dual .* dz, w + α_primal .* dw, s + α_dual .* ds, indices)
            μ̃ = μ_aff * (μ_aff/μ)^2 
        # ================================================================================================================ #

        # ================================================================================================================ #
        # Corrector step: Solve Newton system with the corrector right hand side. Solve as if you wanted to direct the 
        #                 method in the center of the central path.
        # ---------------------------------------------------------------------------------------------------------------- #
            @views @. res_xz[indices.normal] = μ̃ - (dx[indices.normal] * dz[indices.normal])
            @views @. res_xz[indices.box] = μ̃ - (dx[indices.box] * dz[indices.box])
            @views @. res_ws[indices.box] = μ̃ - (dw[indices.box] * ds[indices.box])
            # ============================================================================================================ #
            # Solve the Newton system with the predictor right hand side -> Optimistic view, solve as if you wanted to 
            #                                                               solve the original problem in 1 iteration.
            # ------------------------------------------------------------------------------------------------------------ #
            fill!(res_p, zero(T)) 
            fill!(res_d, zero(T))
            fill!(res_u, zero(T))
            
            CG_solving_elapsed += @elapsed begin
                instability, inneriter_corr = normal_eq_solve!(steps_c, cg_solver, opN_Reg, Pinv, vars, res, indices, krylov_tol)
            end

            if instability == true          # Checking if the matrix is too ill-conditioned. Mitigate it.
                if retry_c < max_tries
                    @printf("The system is re-solved, due to bad conditioning of corrector.\n")
                    δ = δ * 100
                    ρ = ρ * 100
                    iter = iter - 1
                    retry_c = retry_c + 1
                    μ = μ_prev
                    reg_limit = min(reg_limit*10, tol)
                    continue;
                else
                    @printf("The system matrix is too ill-conditioned.\n")
                    break;
                end
            end
            retry_c = zero(Int)
            # ============================================================================================================ #
            @. dx += dx_c
            @. dy += dy_c
            @. dz += dz_c
            @. dw += dw_c
            @. ds += ds_c
            iter > 1 && (inneriter_jump .= [inneriter_pred, inneriter_corr] .- inneriter)
            inneriter .= [inneriter_pred, inneriter_corr]
            sum_inner_iter += inneriter 
            # ================================================================================================================================================================================================================================ #
            # Compute the new iterate:
            # Determine primal and dual step length. Calculate "step to the boundary" αmax_x and αmax_z. 
            # Then choose 0 < τ < 1 heuristically, and set step length = τ * step to the boundary.
            # ---------------------------------------------------------------------------------------------------------------- #
            if n_IJ > 0
                α_primal, α_dual = stepsize_in_orthant(vars, steps, indices)
            else
                α_primal = 1             # If we have no inequality constraints, Newton method is exact -> Take full step.
                α_dual = 1
            end
            # ================================================================================================================================================================================================================================ #
            # Make the step.
            # ---------------------------------------------------------------------------------------------------------------- #
            @. x = x + α_primal * dx 
            @. y = y + α_dual * dy 
            @. z = z + α_dual * dz
            @. w = w + α_primal * dw
            @. s = s + α_dual * ds
            if n_IJ > 0      # Only if we have non-negativity constraints.
                μ_prev = μ
                μ = compute_μ(vars, indices)
                μ_rate = abs((μ - μ_prev) / max(μ, μ_prev))
            end
            # ================================================================================================================ #
            
            # ================================================================================================================ #
            # Computing the new non-regularized residuals. If the overall error is decreased, for the primal and dual 
            # residuals, we accept the new estimates for the Lagrange multipliers and primal optimal solution respectively.
            # If not, we keep the estimates constant.
            # ---------------------------------------------------------------------------------------------------------------- #
            # new_nr_res_p = b - opA*x
            mul!(new_nr_res_p, opA, x, -one(T), zero(T))
            @. new_nr_res_p += b                             # Non-regularized dual residual
            # new_nr_res_d = opQ*x - opAT*y - z + s + c
            mul!(new_nr_res_d, opQ, x)
            mul!(new_nr_res_d, opAT, y, -one(T), one(T))
            @. new_nr_res_d -= z
            @. new_nr_res_d += s
            @. new_nr_res_d += c
            cond_bool = (0.95*norm(nr_res_p) > norm(new_nr_res_p))
            if cond_bool
                λ .= y
                if n_IJ > 0
                    δ = max(reg_limit, δ * (1 - μ_rate))  
                else
                    δ = max(reg_limit, δ * 0.1);                    # In this case, IPM not active -> Standard PMM (heuristic)      
                end
            else
                if n_IJ > 0
                    δ = max(reg_limit, δ * (1 - 0.666 * μ_rate))    # Slower rate of decrease, to avoid losing centrality.       
                else
                    δ = max(reg_limit, δ * 0.5);                    # In this case, IPM not active -> Standard PMM (heuristic)      
                end
                no_dual_update = no_dual_update + 1
            end
            
            cond_bool = (0.95*norm(nr_res_d) > norm(new_nr_res_d))
            if cond_bool
                ζ .= x
                if n_IJ > 0
                    ρ = max(reg_limit, ρ * (1 - μ_rate)) 
                else
                    ρ = max(reg_limit, ρ * 0.1)                     # In this case, IPM not active -> Standard PMM (heuristic)        
                end
            else
                if n_IJ > 0
                    ρ = max(reg_limit, ρ * (1 - 0.666 * μ_rate))    # Slower rate of decrease, to avoid losing centrality.     
                else
                    ρ = max(reg_limit,ρ * 0.5)                      # In this case, IPM not active -> Standard PMM (heuristic)    
                end
                no_primal_update = no_primal_update + 1
            end
            # ================================================================================================================
            
            # ================================================================================================================#
            # Print iteration output.  
            # ---------------------------------------------------------------------------------------------------------------- #
            pres_inf = norm(new_nr_res_p)/max(100,norm(b))
            dres_inf = norm(new_nr_res_d)/max(100,norm(c))  
            print_output(pl, pc, method_P, iter , pres_inf, dres_inf, μ, inneriter, krylov_tol, α_primal, α_dual, ρ, δ);

        end
        # ================================================================================================================ #

        # Update the history dictionary
        push!(history["iter"], iter)
        push!(history["primal_feasibility"], pres_inf)
        push!(history["dual_feasibility"], dres_inf)
        push!(history["optimality_gap"], μ)
        if pc == true
            push!(history["inner_iter_predictor"], inneriter[1])
            push!(history["inner_iter_corrector"], inneriter[2])
        else
            push!(history["inner_iter_predictor"], inneriter)
            push!(history["inner_iter_corrector"], 0)
        end
        push!(history["krylov_tol"], krylov_tol)
        push!(history["rho"], ρ)
        push!(history["delta"], δ)

        if (type_method_P <: method_Nystrom)
            push!(history["rank"], method_P.sketchsize)
        elseif (type_method_P <: method_PartialCholesky)
            push!(history["rank"], method_P.k_steps)
        elseif (type_method_P <: method_NoPreconditioner)
            push!(history["rank"], 0)
        end
        push!(history["construct_precond_elapsed"], construct_precond_elapsed)
        push!(history["CG_solving_elapsed"], CG_solving_elapsed)
        push!(history["cumulative_time"], cumulative_time)
        construct_precond_elapsed = zero(T)
        CG_solving_elapsed = zero(T)

    end # end of while loop

    # The IPM has terminated because the solution accuracy is reached or the maximum number 
    # of iterations is exceeded, or the problem under consideration is infeasible. Print result.  
    pl >= 1 ? begin
    @printf("iterations: %4d\n", iter)
    @printf("primal feasibility: %8.2e\n", norm(opA*x-b)/max(100,norm(b)))
    @printf("dual feasibility: %8.2e\n", norm(opAT*y+z-s-c-opQ*x)/max(100,norm(c)))
    isempty(indices.box) ? nothing : @printf("upper bound: %8.2e\n", norm(u-x[indices.box]-w[indices.box]))
    @printf("complementarity: %8.2e\n", compute_μ(vars, indices))
    (pc == true) ? @printf("sum of inner iterations: %4d (predictor) %4d (corrector)\n", sum_inner_iter[1], sum_inner_iter[2]) : nothing
    end : nothing
    return history, opt, vars
end