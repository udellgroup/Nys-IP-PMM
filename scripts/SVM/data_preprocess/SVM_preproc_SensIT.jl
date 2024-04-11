include(scriptsdir("SVM", "data_preprocess", "SVM_data_preproc_utils.jl"))

features_train, labels_train = read_svm_compressed_example(datadir("SVM", "combined_scale.bz2"));
X_train = hcat(features_train...)
features_test, labels_test = read_svm_compressed_example(datadir("SVM", "combined_scale.t.bz2"));
X_test = hcat(features_test...)

X = hcat(X_train, X_test)
y = vec([class < 3 ? 1.0 : -1.0 for class in vcat(labels_train, labels_test)])
