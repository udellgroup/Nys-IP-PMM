using DataFrames
using MLDatasets 

trainset = MLDatasets.CIFAR10(Tx=Float64, split=:train);
testset = MLDatasets.CIFAR10(Tx=Float64, split=:test);

train_size = length(trainset.targets)
test_size = length(testset.targets)
n = train_size + test_size
m = 32 * 32 * 3

X_train = reshape(trainset.features, (m, train_size))
X_test = reshape(testset.features, (m, test_size))
X = hcat(X_train, X_test)
y = vec([class < 5 ? 1.0 : -1.0 for class in vcat(trainset.targets, testset.targets)])

