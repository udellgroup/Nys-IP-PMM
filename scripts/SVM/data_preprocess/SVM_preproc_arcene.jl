using ZipFile
include(scriptsdir("SVM", "data_preprocess", "SVM_data_preproc_utils.jl"))

data_path = datadir("SVM", "arcene.zip")
# Extract the contents of the compressed folder
zarchive = ZipFile.Reader(data_path)
traindata_index = findfirst(f -> (f.name == "ARCENE/arcene_train.data"), zarchive.files)
X = get_dense_features(zarchive.files[traindata_index]; sep_val=" ")

trainlabels_index = findfirst(f -> (f.name == "ARCENE/arcene_train.labels"), zarchive.files)
lines = readlines(zarchive.files[trainlabels_index])
y = parse.(Float64, lines)
