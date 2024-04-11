using ZipFile
using CodecZlib
using CSV, DataFrames
using Tables

include(scriptsdir("SVM", "data_preprocess", "SVM_data_preproc_utils.jl"))

function convert_to_class(str)
    if str == "BRCA"
        return 1.0
    elseif str == "KIRC"
        return 2.0
    elseif str == "COAD"
        return 3.0
    elseif str == "LUAD"
        return 4.0
    elseif str == "PRAD"
        return 5.0
    else
        error("Invalid class label.")
    end
end

data_path = datadir("SVM", "RNASeq", "TCGA-PANCAN-HiSeq-801x20531")
df_features = CSV.read(joinpath(data_path, "data.csv"), DataFrame)
df_labels = CSV.read(joinpath(data_path, "labels.csv"), DataFrame)
features = Tables.matrix(df_features[:,2:end])
X = Vector{Vector{Float64}}(undef, 0)
for row in eachrow(features)
    push!(X, convert.(Float64, row))
end
X = hcat(X...)
labels = convert_to_class.(df_labels[:,2])
y = vec([class ≤ 2.0 ? 1.0 : -1.0 for class in labels])
