using CodecBzip2
using SparseArrays

##
function read_svm_example(svm_file::String; T::DataType=Float64)
    lines = readlines(svm_file)
	lines = chomp.(lines)
	features = Vector{Vector{T}}(undef, 0)
	labels = Vector{Int}(undef, 0)
	for line in lines
		fields = split(line, " ")
        push!(labels, parse(Int, fields[1]))
		col = Vector{T}(undef, 0)
        for i in 2:(length(fields)-1)
            _ , feature_value = split(fields[i], ":")
            push!(col,parse(Float64, feature_value))
        end
		push!(features, col)
	end

	return features, labels
end

function read_svm_compressed_example(svm_file::String; T::DataType = Float64)
    features = Vector{Vector{T}}(undef, 0)
    labels = Vector{Int}(undef, 0)
    open(Bzip2DecompressorStream, svm_file) do stream
		for line in eachline(stream)
			fields = split(line, " ")
			push!(labels, parse(Int, fields[1]))
			col = Vector{T}(undef, 0)
			for i in 2:(length(fields)-1)
				_, feature_value = split(fields[i], ":")
				push!(col, parse(Float64, feature_value))
			end
			push!(features, col)
		end
    end

    return features, labels
end

function read_svm_compressed_sparse_example(svm_file::String; T::DataType = Float64)
    labels = Vector{T}(undef, 0)
	I = Vector{Int}(undef, 0)
	J = Vector{Int}(undef, 0)
	V = Vector{T}(undef, 0)
    open(Bzip2DecompressorStream, svm_file) do stream
		for (j, line) in enumerate(eachline(stream))
			fields = split(line, " ")
			push!(labels, parse(T, fields[1]))
			for i in 2:(length(fields)-1)
				feature_index, feature_value = split(fields[i], ":")
				push!(I, parse(Int, feature_index))
				push!(J, j)
				push!(V, parse(Float64, feature_value))
			end
		end
    end
	features = sparse(I, J, V);

    return features, labels
end

"""
    get_dense_features(file::Any)

file: needs readlines() defined.
sep_val: the separator between values in each line
"""
function get_dense_features(file::Any; sep_val::String = " ", T::DataType=Float64)
    lines = readlines(file)
    lines = chomp.(lines)
    features = Vector{Vector{T}}(undef, 0)
    for line in lines   # Each line is the feature vector of a sample.
        feature_vals = split(line, sep_val)
        col = Vector{T}(undef, 0)
        for val in feature_vals[1:end-1]
            push!(col, parse(Float64, val))
        end
        push!(features, col)
    end
    features = hcat(features...)
    return features
end

"""
    get_sparse_features(file::Any, m::Int)

file: needs readlines() defined.
m: the number of features
sep_val: the separator between feature values in each line
sep_ind: the separator between index and value in each feature
"""
function get_sparse_features(file::Any, m::Int; sep_val::String = " ", sep_ind::String = ":", T::DataType=Float64)
    lines = readlines(file)
    lines = chomp.(lines)
	Is = Vector{Int}(undef, 0)
	Js = Vector{Int}(undef, 0)
	Vs = Vector{T}(undef, 0)
	for sample_index in eachindex(lines)
		fields = split(lines[sample_index], sep_val)
		isempty(fields[end]) ? pop!(fields) : nothing
		for ind_val in fields
			feature_index, feature_value = split(ind_val, sep_ind)
			push!(Is, parse(Int, feature_index))
			push!(Js, sample_index)
			push!(Vs, parse(Float64, feature_value))
		end
	end
	n = length(lines)
	features = sparse(Is, Js, Vs, m, n)
    return features
end