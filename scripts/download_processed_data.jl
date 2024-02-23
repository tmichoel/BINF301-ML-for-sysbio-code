using DrWatson
@quickactivate "BINF301-code"

using JuliaHub
using DataSets

# Download the processed TCGA breast cancer data from JuliaHub
JuliaHub.download_dataset(JuliaHub.dataset(("tmichoel-1", "TCGA_BRCA")), datadir("processed","TCGA_BRCA"))

# Download the processed CCLE data from JuliaHub
JuliaHub.download_dataset(JuliaHub.dataset(("tmichoel-1", "CCLE")), datadir("processed","CCLE"))

# Download the processed Mouse V1/ALM SMART-seq data from JuliaHub
JuliaHub.download_dataset(JuliaHub.dataset(("tmichoel-1", "Mouse_V1_ALM")), datadir("processed","Mouse_V1_ALM"))