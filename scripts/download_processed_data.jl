using DrWatson
@quickactivate "BINF301-code"

using JuliaHub
using DataSets

# Download the processed TCGA breast cancer data from JuliaHub
ds = JuliaHub.dataset(("tmichoel-1", "TCGA_BRCA"))
JuliaHub.download_dataset(ds,datadir("processed","TCGA_BRCA"))