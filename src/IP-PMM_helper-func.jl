###########################################################################################
##                          Helper Functions for IP_PMM_bdd()                                ##
###########################################################################################

function dim_checking(m::Int, n::Int, c, opQ, b, u, indices::IPMIndices)
    ind_I = @views indices.normal    # indices of positive unbounded variables xᴵ ≥ 0
    ind_J = @views indices.box    # indices of box-constrained variables 0 ≤ xᴶ ≤ u
    ind_F = @views indices.free    # indices of free variables
    n_I = length(ind_I)         # number of positive unbounded variables (= |I|)
    n_J = length(ind_J)         # number of box-constrained variables (= |J|)
    n_F = length(ind_F)         # number of free variables (= |F|)
    @assert (length(c) == n) && (length(b) == m) && (size(opQ) == (n, n)) "Problem dimension mismatch."
    @assert ((n_I + n_J + n_F) == n) "The sum of cardinality |I| + |J| + |F| must be the number of variables n."
    @assert (n_J == length(u)) "The number of box-constrained variables |J| must equal to the length of u."
    @assert isdisjoint(ind_I, ind_J) && isdisjoint(ind_I, ind_F) && isdisjoint(ind_J, ind_F) "The sets of indices I, J, F must be disjoint."
    @assert (setdiff([i for i in 1:n], vcat(ind_I, ind_J, ind_F)) == []) "The sets of indices I, J, F must cover all variables."
    @assert all(isfinite.(u)) @error("Upper bound vector u has Inf entries.")
    return nothing
end

function obtain_initial_point!(initial_point::IPMVariables{T}, input::IPMInput{T};
                               cg_solver::CgSolver{T, T, Vector{T}} = nothing, 
                               sketchsize::Int = -1, 
                               printlevel::Int = 1, krylov_tol::Real = 1e-6) where {T}
    # Unwrap input
    nrow, ncol = input.nrow, input.ncol
    opA = @views input.opA
    b = @views input.b
    c = @views input.c
    u = @views input.u
    opQ = @views input.opQ
    
    # If not given cg_solver, construct one.
    isnothing(cg_solver) && (cg_solver = CgSolver(nrow, nrow, T)) 

    # If not given sketchsize, set it to be 1/4 of ncol
    (sketchsize < 0) && (sketchsize = min(floor(Int, nrow/4), 100))
    
    # Unwrap initial_point
    x = @views initial_point.x
    y = @views initial_point.y
    z = @views initial_point.z
    w = @views initial_point.w
    s = @views initial_point.s

    # Unwrap indices
    indices = @views input.indices
    free_vars = @views indices.free
    posunbdd_vars = @views indices.normal
    box_vars = @views indices.box
    pos_vars = union(posunbdd_vars, box_vars)
    n = length(x)
    n_J = length(box_vars)
    n_IJ = length(pos_vars)

    xᴵᴶ = @views x[pos_vars]
    xᴵ = @views x[posunbdd_vars]
    xᴶ = @views x[box_vars]
    zᴵᴶ = @views z[pos_vars]
    zᴵ = @views z[posunbdd_vars]
    zᴶ = @views z[box_vars]
    wᴶ = @views w[box_vars]
    sᴶ = @views s[box_vars]
    # =================================================================================================================== #
    # Use PCG to solve two least-squares problems for efficiency (along with the Nystrom preconditioner). 
    # ------------------------------------------------------------------------------------------------------------------- #
    # m = size(opA, 1)
    δ = 10.0

    opAT = @views opA'
    opAAT = opA * opAT
    N̂ = NystromSketch(opAAT, sketchsize)
    Pinv = NystromPreconditionerInverse(N̂, δ)

    # x = ū/2 + Aᵀ(AAᵀ + δI)⁻¹(b - A ū/2) where ū is the embedding of u into Rⁿ
    ū = zeros(T, n)
    ū[box_vars] .= u
    RHS_vec = b - 0.5 * (opA * ū)
    cg!(cg_solver, opAAT + δ * opEye(nrow, nrow), RHS_vec; rtol=krylov_tol, itmax=min(1000, nrow), M = Pinv)
    x .= opAT * cg_solver.x
    @. xᴶ += 0.5 * u
    cgiter = cg_solver.stats.niter

    # y = (AAᵀ + δI)⁻¹A(c + Qx)
    RHS_vec = opA * (c + opQ * x)
    cg!(cg_solver, opAAT + δ * opEye(nrow, nrow), RHS_vec; rtol=krylov_tol, itmax=min(1000, nrow), M = Pinv)
    copyto!(y, cg_solver.x)
    cgiter += cg_solver.stats.niter

    # zᴵ = E_I (c + Qx - Aᵀy), zᴶ = 0.5 * E_J (c + Qx - Aᵀy), zF = 0
    z .= c + opQ * x - opAT * y
    @. zᴶ *= 0.5

    # sᴶ = -zᴶ
    @. sᴶ = -zᴶ

    # wᴶ = u - xᴶ
    @. wᴶ = u - xᴶ

    printlevel ≥ 1 && @printf("CG iterations for starting point: %4d \n", cgiter)
    # =================================================================================================================== %
    
    # Ensure sufficient magnitude of xᴵᴶ, zᴵᴶ, wᴶ, sᴶ
    (norm(xᴵ) ≤ 1e-4) ? xᴵ .= 0.1 : nothing      # 0.1 is chosen arbitrarily
    (norm(xᴶ) ≤ 1e-4) ? xᴶ .= 0.1 : nothing      # 0.1 is chosen arbitrarily
    (norm(wᴶ) ≤ 1e-4) ? wᴶ .= 0.1 : nothing      # 0.1 is chosen arbitrarily
    (norm(zᴵ) ≤ 1e-4) ? zᴵ .= 0.1 : nothing      # 0.1 is chosen arbitrarily
    (norm(zᴶ) ≤ 1e-4) ? zᴶ .= 0.1 : nothing      # 0.1 is chosen arbitrarily
    (norm(sᴶ) ≤ 1e-4) ? sᴶ .= 0.1 : nothing      # 0.1 is chosen arbitrarily

    # Ensure positivity of xᴵᴶ, zᴵᴶ, wᴶ, sᴶ
    δ_primal = (n_J == 0) ? max(-1.5 * minimum(xᴵᴶ), 0) : max(-1.5 * minimum(xᴵᴶ), -1.5 * minimum(wᴶ), 0)
    δ_dual = (n_J == 0) ? max(-1.5 * minimum(zᴵᴶ), 0) : max(-1.5 * minimum(zᴵᴶ), -1.5 * minimum(sᴶ), 0)
    temp_product = dot(xᴵᴶ, zᴵᴶ) + dot(wᴶ, sᴶ)
    δ̄_primal = δ_primal + (0.5 * temp_product) / (sum(zᴵᴶ) + sum(sᴶ) + (n_IJ + n_J) * δ_dual)
    δ̄_dual = δ_dual + (0.5 * temp_product) / (sum(xᴵᴶ) + sum(wᴶ) + (n_IJ + n_J) * δ_primal)

    # update x 
    @. xᴵᴶ += δ̄_primal              # xᴵᴶ = xᴵᴶ + δ̄_primal * e_IJ
    # update z
    @. zᴵᴶ += δ̄_dual                # zᴵᴶ = zᴵᴶ + δ̄_dual * e_IJ
    @. z[free_vars] = zero(T)  # zF = 0
    # update w
    @. wᴶ += δ̄_primal               # wᴶ = wᴶ + δ̄_primal * e_J
    @. w[posunbdd_vars] = zero(T)   # wᴵ = 0
    @. w[free_vars] = zero(T)  # wF = 0
    # update s
    @. sᴶ += δ̄_dual                 # sᴶ = sᴶ + δ̄_dual * e_J
    @. s[posunbdd_vars] = zero(T)   # sᴵ = 0
    @. s[free_vars] = zero(T)  # sF = 0 

    return cgiter
end


### Compute μ and σ


function compute_μ(x::AbstractVector{T}, z::AbstractVector{T}, w::AbstractVector{T}, s::AbstractVector{T}, indices::IPMIndices) where {T<:Number}
    ind_I = @views indices.normal
    ind_J = @views indices.box
    xᵀz = dot(x[ind_I], z[ind_I]) + dot(x[ind_J], z[ind_J])
    wᵀs = dot(w[ind_J], s[ind_J])
    return (xᵀz + wᵀs) / (length(ind_I) + 2 * length(ind_J))
end

compute_μ(vars::IPMVariables{T}, indices::IPMIndices) where {T<:Number} = compute_μ(vars.x, vars.z, vars.w, vars.s, indices)



function update_σ(iter::Int, α_primal::Number, α_dual::Number, σmin::Number, σmax::Number)
    if (iter > 1)
        σ = max(1 - α_primal, 1 - α_dual)^5
    else
        σ = 0.5
    end
    σ = min(σ, σmax)
    σ = max(σ, σmin)
    return σ
end

## Step-size computation

function stepsize_in_orthant(vars::IPMVariables{T}, steps::IPMVariables{T}, indices::IPMIndices) where {T}
    x = @views vars.x
    z = @views vars.z
    w = @views vars.w
    s = @views vars.s

    dx = @views steps.x
    dz = @views steps.z
    dw = @views steps.w
    ds = @views steps.s

    n = length(x)
    idx = falses(n)
    idz = falses(n)
    idw = falses(n)
    ids = falses(n)
    @. idx[indices.normal] = (dx[indices.normal] < 0)            # Select all the negative dx's (dz's respectively)
    @. idx[indices.box] = (dx[indices.box] < 0)
    @. idz[indices.normal] = (dz[indices.normal] < 0)
    @. idz[indices.box] = (dz[indices.box] < 0)
    @. idw[indices.box] = (dw[indices.box] < 0)
    @. ids[indices.box] = (ds[indices.box] < 0)

    αmax_primal = minimum(vcat(one(T), -x[idx] ./ dx[idx], -w[idw] ./ dw[idw]))
    αmax_dual = minimum(vcat(one(T), -z[idz] ./ dz[idz], -s[ids] ./ ds[ids]))
    τ = 0.995
    α_primal = τ * αmax_primal
    α_dual = τ * αmax_dual

    return α_primal, α_dual
end


"""
    VectorTransposeOperator(y)

Construct LinearOperator yᵀ for a vector y.
"""
function VectorTransposeOperator(y::AbstractVector{T}) where T <: Number
    return LO.LinearOperator(T, 1, length(y), false, false,
                            (res, v) -> copyto!(res, dot(y,v)),
                            (res, v) -> copyto!(res, v .* y),
                            (res, v) -> copyto!(res, v .* conj(y)))
end