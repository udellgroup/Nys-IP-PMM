using ZipFile
include(scriptsdir("SVM", "data_preprocess", "SVM_data_preproc_utils.jl"))

data_path = datadir("SVM", "dexter.zip")
# Extract the contents of the compressed folder
zarchive = ZipFile.Reader(data_path)
traindata_index = findfirst(f -> (f.name == "DEXTER/dexter_train.data"), zarchive.files)
X = get_sparse_features(zarchive.files[traindata_index], 20_000)

trainlabels_index = findfirst(f -> (f.name == "DEXTER/dexter_train.labels"), zarchive.files)
lines = readlines(zarchive.files[trainlabels_index])
y = parse.(Float64, lines)