################################################################################################
### This script downloads the datasets used in the SVM experiments.
### Run this script before running "SVM_run_tests.jl".
### It might take a while to download all the datasets.
################################################################################################
using CodecZlib
using DrWatson
using Downloads
using Tar
using ZipFile

# Function to download datasets from url
"""
Download datasets from url to download_path. 

    download_from_url(url::String, download_path::String)

If the file already exists, it will not be downloaded again.
If the file does not exist, it will be downloaded to the specified path.
"""
function download_from_url(url::String, download_path::String)
    if isfile(download_path)
        println("File $download_path already exists.")
    else 
        (!isdir)(dirname(download_path)) && mkpath(dirname(download_path)) 
	println("Downloading $url to $download_path...")
    Downloads.download(url, download_path)
    println("Downloaded $url to $download_path successfully.")
    end
    return nothing
end

########################################
## UCI ML Repository: Arcene
########################################
println("Downloading Arcene dataset")
url = "https://archive.ics.uci.edu/static/public/167/arcene.zip"  # The URL of the compressed folder
download_path = datadir("SVM", "arcene.zip")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)

########################################
## UCI ML Repository: Dexter
########################################
println("Downloading Dexter dataset")
url = "https://archive.ics.uci.edu/static/public/168/dexter.zip"  # The URL of the compressed folder
download_path = datadir("SVM", "dexter.zip")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)

########################################
## UCI ML Repository: RNASeq
########################################
println("Downloading RNASeq dataset")
url = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
download_path = datadir("SVM", "RNASeq.zip")  # Path to save the downloaded compressed folder
extracted_file = datadir("SVM", "RNASeq.tar.gz")
download_from_url(url, download_path)
zarchive = ZipFile.Reader(download_path)
for file in zarchive.files
    isdir(dirname(extracted_file)) ? nothing : mkdir(dirname(extracted_file))
    open(extracted_file, "w") do io
        write(io, read(file))
    end
end
data_path = datadir("SVM", "RNASeq.tar.gz")
!isdir(datadir("SVM", "RNASeq")) && Tar.extract(GzipDecompressorStream(open(data_path,"r")), datadir("SVM", "RNASeq"))

########################################
## STL-10
########################################
println("Downloading STL-10 dataset")
url = "http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz"
download_path = datadir("SVM", "STL10.tar.gz")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)
!isdir(datadir("SVM", "STL10")) && Tar.extract(GzipDecompressorStream(open(download_path,"r")), datadir("SVM", "STL10"))

########################################
## Sector
########################################
println("Downloading Sector dataset")
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.scale.bz2"
download_path = datadir("SVM", "sector.scale.bz2")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.t.scale.bz2"
download_path = datadir("SVM", "sector.t.scale.bz2")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)


########################################
## SensIT
########################################
println("Downloading SensIT dataset")
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.bz2"
download_path = datadir("SVM", "combined_scale.bz2")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined_scale.t.bz2"
download_path = datadir("SVM", "combined_scale.t.bz2")  # Path to save the downloaded compressed folder
download_from_url(url, download_path)