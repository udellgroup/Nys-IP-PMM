using DrWatson
@quickactivate "."
using LinearAlgebra
using Random
using LinearOperators
using BenchmarkTools
const LO = LinearOperators

include(srcdir("IP-PMM_structs.jl"))
include(srcdir("normal_equations.jl"))

# Script to compare the two approaches
function compare_approaches()
    # Generate a random matrix A
    Random.seed!(42)
    m, n = 3073, 6000  # Size of the matrix (m x n)
    A = randn(m, n)
    diagD = abs.(randn(n))

    opA = LO.LinearOperator(A)
    opN1 = opNormalEquations(opA)
    opN2 = opNormalEquations(opA)

    # Initialize the diagonal
    opN1.D.diag .= diagD
    opN2.D.diag .= diagD

    # Compute diagonal using the first approach
    @time update_diagN_opN!(opN1, nothing)

    # Compute diagonal using the second approach
    @time update_diagN_opN!(opN2, A)

    # Compare the results
    diff = maximum(abs.(opN1.diagN .- opN2.diagN))
    println("Maximum difference between approaches: $diff")

    # Benchmark the performance of each approach
    println("\nBenchmarking first approach:")
    @btime update_diagN_opN!($opN1, nothing)

    println("\nBenchmarking second approach:")
    @btime update_diagN_opN!($opN2, $A)
end

# Run the comparison
compare_approaches()


# ##
# # Generate a random matrix A
# Random.seed!(42)
# m, n = 3073, 6000  # Size of the matrix (m x n)
# A = randn(m, n)
# diagD = abs.(randn(n))
# N = A * Diagonal(diagD) * A'

# opA = LO.LinearOperator(A)
# opN1 = opNormalEquations(opA)
# opN2 = opNormalEquations(opA)

# # Initialize the diagonal
# opN1.D.diag .= diagD
# opN2.D.diag .= diagD