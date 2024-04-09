const LO = LinearOperators


############################################
############################################
## Normal Equations Preconditioner
############################################
############################################

############################################
##  No Preconditioner
############################################
"""
Method of preconditioner: No preconditioner

    struct method_NoPreconditioner

"""
struct method_NoPreconditioner end

allocate_preconditioner(method_P::method_NoPreconditioner, opNreg) = I     # Identity operator
function update_preconditioner!(method_P::method_NoPreconditioner, Pinv, opN_Reg, adaptive_info...)
    Pinv = I
    return nothing 
end

############################################
##  Nystrom Preconditioner
############################################
include(srcdir("NysPreconditioner.jl"))

############################################
##  Partial Cholesky Preconditioner
############################################
include(srcdir("PartialCholesky.jl"))
