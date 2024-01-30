### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 997bb1b8-865a-4504-be9d-9aa7daa554b5
using DrWatson

# ╔═╡ 701c1990-fffd-4d74-835d-272092d6c929
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ 39e893c4-1cde-4bbd-b464-72292513973b
begin
	using DataFrames
	using CSV
	using DataSets
	using Statistics
	using StatsBase
	using StatsPlots, LaTeXStrings
	using FreqTables
	# using MultivariateStats
	# using LinearAlgebra
	# using Random
	using Clustering
	# using Distances
	using PlutoUI
	using Printf
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

# ╔═╡ 86eca1aa-0606-4d69-9d83-7ad8a9189038
md"
## Create clinical outcome groupings

- Triple negative tumours (yes/no)
- Stage
"

# ╔═╡ 10343544-6ff4-435b-96f2-6e9ca6dc9d72
triple_neg = df_clin.:"ER Status" .== "Negative" .&& df_clin.:"PR Status" .== "Negative" .&& df_clin.:"HER2 Final Status" .== "Negative"

# ╔═╡ cfeb4e9f-81f5-4e7b-8d60-a049342fa104
# ╠═╡ disabled = true
#=╠═╡
triple_neg = df_clin.:"ER Status" .== "Negative"
  ╠═╡ =#

# ╔═╡ f940f078-ea2d-4a7f-b55b-9e09bab920d4
begin
	stage = zeros(Int16,nrow(df_clin))
	stage[map(x -> ∈(x, ["Stage I", "Stage IA", "Stage IB"]), 
		df_clin.:"AJCC Stage")] .= 1
	stage[map(x -> ∈(x, ["Stage II", "Stage IIA", "Stage IIB"]), 
		df_clin.:"AJCC Stage")] .= 2
	stage[map(x -> ∈(x, ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"]), 
		df_clin.:"AJCC Stage")] .= 3
	stage[df_clin.:"AJCC Stage" .== "Stage IV"] .= 4
end

# ╔═╡ 3c8ac526-e0ce-45f2-b634-5aeb53f13dc9
md"
## Explore expression data

In cluster analysis we often standardize features (genes in our case) to give all of them equal weight in the Euclidean distance measure between observations (breast tumours in our case), see Elements of Statistical Learning section ...

In the figure below, we see a histogram of standard deviations of all genes. Move the slider to decide on a standard deviation cutoff where only genes with standard deviation above the cutoff are retained.

"

# ╔═╡ 0ed193aa-9c4a-4f06-87b4-d81fa1252be5
@bind sdcut Slider(0:0.1:2)

# ╔═╡ e53173a6-c32e-4ecb-997f-941803b20600
begin
	sd = map(x -> std(x), eachcol(df_expr));
	ng = sum(sd .> sdcut)
	histogram(sd, xlabel="Standard deviation", label="")
	vline!([sdcut],linewidth=2,label="")
	annotate!(2.5,1700,@sprintf("Standard deviation cutoff: %1.1f", sdcut))
	annotate!(2.5,1400,@sprintf("Number of selected genes: %d", ng))
end

# ╔═╡ 5519b676-f50d-4f19-b4f5-40b43cbbbad2
df_expr_z = mapcols(zscore, df_expr);

# ╔═╡ 7af16ff6-46e9-4410-99fd-8a3521f98dc3
md"## K-means clustering

We use K-means clustering with $K=2$ to analyze the overlap between expression clusters and triple-negative status. 
"

# ╔═╡ 219bf88d-1bf7-4d27-a7c8-a71827f31876
km_clust2 = kmeans(Matrix(df_expr_z[:,sd.>sdcut])',2);

# ╔═╡ 1d504701-65e9-443e-887d-bc77d8148b87
md"Compute the overlap count for each cluster with ER positive and negative samples."

# ╔═╡ 4afd5aa6-34c2-41ae-b458-1d98396def5f
freqtable(km_clust2.assignments,triple_neg)

# ╔═╡ f72e3abb-1ead-4e31-b69a-bab7a3292ba4
md"
Now use K-means clustering with $K=4$ to analyze the overlap between gene expression clusters and tumour stage, using only samples with non-zero stage.
"

# ╔═╡ 40a718d0-510f-4e30-8b37-43f05a0d284c
km_clust4 = kmeans(Matrix(df_expr_z[stage.>0,sd.>sdcut])',4);

# ╔═╡ 376ebe3c-89ee-4818-a04d-bfa7133252d1
freqtable(km_clust4.assignments, stage[stage .> 0])

# ╔═╡ 3dabdc41-e64c-4b23-a0ad-0dc8c43b8d16
md"
## Test clustering theory
"

# ╔═╡ 246144f1-b43a-410f-a235-4403ba314f93
gmax = argmax(sd)

# ╔═╡ 56a428ce-01bd-4ad2-8d72-d16ec6f33f1e
gmin = argmin(sd)

# ╔═╡ 1af90006-e58c-43ce-be62-2d6a4e568d3e
scatter(df_expr[:,gmin],df_expr[:,gmax])

# ╔═╡ 7d3a7b6e-accf-452c-b7b6-7681fc590893
mapcols(zscore,X)

# ╔═╡ bad52ca6-abed-48f9-9ad8-5ab47973da9d
km_noz = kmeans(Matrix(df_expr[:, [gmin, gmax]])',2);

# ╔═╡ 2e716cab-c7f1-4120-84c4-447d51d8ce2c
km_z = kmeans(Matrix(df_expr_z[:, [gmin, gmax]])',2);

# ╔═╡ a29a090b-28ca-44bf-a15e-f9bc9f242b13
df_expr[:, [gmin, gmax]]

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
# ╟─86eca1aa-0606-4d69-9d83-7ad8a9189038
# ╠═10343544-6ff4-435b-96f2-6e9ca6dc9d72
# ╠═cfeb4e9f-81f5-4e7b-8d60-a049342fa104
# ╠═f940f078-ea2d-4a7f-b55b-9e09bab920d4
# ╟─3c8ac526-e0ce-45f2-b634-5aeb53f13dc9
# ╠═0ed193aa-9c4a-4f06-87b4-d81fa1252be5
# ╠═e53173a6-c32e-4ecb-997f-941803b20600
# ╠═5519b676-f50d-4f19-b4f5-40b43cbbbad2
# ╟─7af16ff6-46e9-4410-99fd-8a3521f98dc3
# ╠═219bf88d-1bf7-4d27-a7c8-a71827f31876
# ╟─1d504701-65e9-443e-887d-bc77d8148b87
# ╠═4afd5aa6-34c2-41ae-b458-1d98396def5f
# ╟─f72e3abb-1ead-4e31-b69a-bab7a3292ba4
# ╠═40a718d0-510f-4e30-8b37-43f05a0d284c
# ╠═376ebe3c-89ee-4818-a04d-bfa7133252d1
# ╟─3dabdc41-e64c-4b23-a0ad-0dc8c43b8d16
# ╠═246144f1-b43a-410f-a235-4403ba314f93
# ╠═56a428ce-01bd-4ad2-8d72-d16ec6f33f1e
# ╠═1af90006-e58c-43ce-be62-2d6a4e568d3e
# ╠═7d3a7b6e-accf-452c-b7b6-7681fc590893
# ╠═bad52ca6-abed-48f9-9ad8-5ab47973da9d
# ╠═2e716cab-c7f1-4120-84c4-447d51d8ce2c
# ╠═a29a090b-28ca-44bf-a15e-f9bc9f242b13
