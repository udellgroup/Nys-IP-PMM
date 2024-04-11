include(scriptsdir("SVM", "data_preprocess", "SVM_data_preproc_utils.jl"))

features_train, labels_train = read_svm_compressed_sparse_example(datadir("SVM", "sector.scale.bz2"));
features_test, labels_test = read_svm_compressed_sparse_example(datadir("SVM", "sector.t.scale.bz2"));

X = hcat(features_train, features_test)
y = vec([class ≤ 50 ? 1.0 : -1.0 for class in vcat(labels_train, labels_test)])
