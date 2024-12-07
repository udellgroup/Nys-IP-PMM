using DrWatson
@quickactivate "."
using LinearAlgebra
using Random
using LinearOperators
using BenchmarkTools
const LO = LinearOperators

include(srcdir("IP-PMM_structs.jl"))
include(srcdir("normal_equations.jl"))

# Define the second approach (row-wise computation of diagonal)
function update_diagN_rowwise_opN!(opN::opNormalEquations{T}, A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @views diagN = opN.diagN
    @views diagD = opN.D.diag
    @views tmp = opN.tmp

    @inbounds @simd for i in 1:n
        tmp .= A[i, :]
        diagN[i] = dot(tmp, tmp) * diagD[i]
    end
    return nothing
end

# Script to compare the two approaches
function compare_approaches()
    # Generate a random matrix A
    Random.seed!(42)
    m, n = 3073, 60000  # Size of the matrix (m x n)
    A = randn(m, n)

    opA = LO.LinearOperator(A)
    opN1 = opNormalEquations(opA)
    opN2 = opNormalEquations(opA)

    # Compute diagonal using the first approach
    @time update_diagN_opN!(opN1)

    # Compute diagonal using the second approach
    @time update_diagN_rowwise_opN!(opN2, A)

    # Compare the results
    diff = maximum(abs.(opN1.diagN .- opN2.diagN))
    println("Maximum difference between approaches: $diff")

    # Benchmark the performance of each approach
    println("\nBenchmarking first approach (update_diagN_opN!):")
    @btime update_diagN_opN!($opN1)

    println("\nBenchmarking second approach (update_diagN_rowwise_opN!):")
    @btime update_diagN_rowwise_opN!($opN2, $A)
end

# Run the comparison
compare_approaches()
