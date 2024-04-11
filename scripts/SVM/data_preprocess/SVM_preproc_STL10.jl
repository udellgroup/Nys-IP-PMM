using MAT

file_train = matread(datadir("SVM", "STL10", "stl10_matlab", "train.mat"))
file_test = matread(datadir("SVM", "STL10", "stl10_matlab", "test.mat"))
X = Vector{Vector{Float64}}(undef, 0)
for row in eachrow(file_train["X"])
    push!(X, convert.(Float64, row))
end
for row in eachrow(file_test["X"])
    push!(X, convert.(Float64, row))
end
X = hcat(X...)
y = vec([class < 5 ? 1.0 : -1.0 for class in vcat(file_train["y"], file_test["y"])])
