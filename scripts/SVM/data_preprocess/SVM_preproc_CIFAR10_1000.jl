include(scriptsdir("SVM", "data_preprocess", "SVM_preproc_CIFAR10.jl"))

size_test = 1_000
X = X[:, 1:size_test]
ylabel = vec([class < 5 ? 1.0 : -1.0 for class in vcat(trainset.targets, testset.targets)])
y = @views ylabel[1:size_test];