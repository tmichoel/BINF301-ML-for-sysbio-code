### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 997bb1b8-865a-4504-be9d-9aa7daa554b5
using DrWatson

# ╔═╡ 701c1990-fffd-4d74-835d-272092d6c929
quickactivate(@__DIR__)

# ╔═╡ 39e893c4-1cde-4bbd-b464-72292513973b
begin
	using DataFrames
	using CSV
	using DataSets
	# using Statistics
	# using StatsBase
	# using StatsPlots, LaTeXStrings
	# using MultivariateStats
	# using LinearAlgebra
	# using Random
	# using Clustering
	# using Distances
	# using PlutoUI
	# using Printf
end

# ╔═╡ 6cb72390-b393-11ee-1a18-af78f69f2995
md"# Cluster analysis of the TCGA BRCA data
## Setup the environment

Activate the BINF301 environment:
"

# ╔═╡ cfab9820-de64-4d20-8354-856bf0c08a62
md"Load packages:"

# ╔═╡ 5b1a8ea9-1044-4662-8da5-6c4f025d7f84
md"
## Load the data

Make sure that you have downloaded the data before running this notebook by executing the script `download_processed_data.jl` in the `scripts` folder. Information to load the data is stored in the `Data.toml` file, which needs to be loaded first:
"

# ╔═╡ 0d42d399-b1fa-41e6-a371-6a014d9b7161
DataSets.load_project!(projectdir("Data.toml"))

# ╔═╡ bc9c853d-558d-4871-a45a-b9a6db78b4ff
md"""
Open the dataset and read each of the files in a DataFrame. This code is adapted from the [DataSets tutorial on JuliaHub](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/).
"""

# ╔═╡ 0a74b8b0-af7f-4c71-a404-fb8a2b3c6404
tree = DataSets.open(dataset("TCGA_BRCA"))

# ╔═╡ 61466b7d-8289-4996-b956-6e231b0953f7
md"
The first file contains clinical data for each sample:
"

# ╔═╡ f4afdd2e-77fd-499a-bf22-e52189401cc4
df_clin = open(Vector{UInt8}, tree["TCGA-BRCA-exp-348-clin.csv"]) do buf
           CSV.read(buf, DataFrame);
       end

# ╔═╡ cdffa0e4-db96-4822-9e59-b23d466cb42c
md"The second file contains gene expression data, with samples (rows) in the same order as in the clinical data:"

# ╔═╡ bff7010b-8638-40ee-99b6-9978d77d2bdc
df_expr = open(Vector{UInt8}, tree["TCGA-BRCA-exp-348-expr.csv"]) do buf
           CSV.read(buf, DataFrame)
       end

# ╔═╡ Cell order:
# ╟─6cb72390-b393-11ee-1a18-af78f69f2995
# ╠═997bb1b8-865a-4504-be9d-9aa7daa554b5
# ╠═701c1990-fffd-4d74-835d-272092d6c929
# ╟─cfab9820-de64-4d20-8354-856bf0c08a62
# ╠═39e893c4-1cde-4bbd-b464-72292513973b
# ╟─5b1a8ea9-1044-4662-8da5-6c4f025d7f84
# ╠═0d42d399-b1fa-41e6-a371-6a014d9b7161
# ╟─bc9c853d-558d-4871-a45a-b9a6db78b4ff
# ╠═0a74b8b0-af7f-4c71-a404-fb8a2b3c6404
# ╟─61466b7d-8289-4996-b956-6e231b0953f7
# ╠═f4afdd2e-77fd-499a-bf22-e52189401cc4
# ╟─cdffa0e4-db96-4822-9e59-b23d466cb42c
# ╠═bff7010b-8638-40ee-99b6-9978d77d2bdc
