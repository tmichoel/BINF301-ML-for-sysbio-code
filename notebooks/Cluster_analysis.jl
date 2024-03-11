### A Pluto.jl notebook ###
# v0.19.38

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
#triple_neg = df_clin.:"ER Status" .== "Negative"
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

# ╔═╡ 7f14c0d2-d32d-4b34-923a-4829ce4e3322
md"
Many functions in Statsbase.jl work with matrices instead of dataframes. Here's the expression data in matrix form:
"

# ╔═╡ bece39e9-8d0d-45a6-b867-5d005062ed1a
X = Matrix(df_expr);

# ╔═╡ 193ded13-8c8e-411a-b6c1-72e3b11d90fc
md"
Create a version of the expression data where each gene is standardized to have mean zero and standard deviation one: 
"

# ╔═╡ 2e8d67e5-a09f-48cb-a12a-49a28a84fab9
X_std = standardize(ZScoreTransform, Matrix(df_expr); dims=1);

# ╔═╡ 7af16ff6-46e9-4410-99fd-8a3521f98dc3
md"## K-means clustering

We use K-means clustering with $K=2$ to analyze the overlap between expression clusters and triple-negative status. 
"

# ╔═╡ 219bf88d-1bf7-4d27-a7c8-a71827f31876
km_clust2 = kmeans(X_std[:,sd.>sdcut]',2);

# ╔═╡ 1d504701-65e9-443e-887d-bc77d8148b87
md"
Compute the overlap count for each cluster with ER positive and negative samples. Do you get different results when using standardized or non-standardized data? Or when you change the standard deviation cutoff for filtering genes?
"

# ╔═╡ 4afd5aa6-34c2-41ae-b458-1d98396def5f
freqtable(km_clust2.assignments,triple_neg)

# ╔═╡ f72e3abb-1ead-4e31-b69a-bab7a3292ba4
md"
Now use K-means clustering with $K=4$ to analyze the overlap between gene expression clusters and tumour stage, using only samples with non-zero stage. Do you get different results when using standardized or non-standardized data? Or when you change the standard deviation cutoff for filtering genes?
"

# ╔═╡ 40a718d0-510f-4e30-8b37-43f05a0d284c
km_clust4 = kmeans(X_std[stage.>0,sd.>sdcut]',4);

# ╔═╡ 376ebe3c-89ee-4818-a04d-bfa7133252d1
freqtable(km_clust4.assignments, stage[stage .> 0])

# ╔═╡ abc8da5a-9192-45c8-84d4-e61d4af5e426
md"
How do you interpret the previous results?
"

# ╔═╡ 3dabdc41-e64c-4b23-a0ad-0dc8c43b8d16
md"
## Finding the \"right\" number of clusters

In the [*Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) (Section 14.3.11) it is suggested to examine the within-cluster dissimilarity $W_K$ as a function of the number of clusters $K$. The optimal value of $K$ is obtained by identifying a \"kink\" in the plot of $W_K$ (or $\log W_K$ to be precise) versus $K$.

After running the `kmeans` function with a certain value of $K$, the within-cluster dissimilarity $W_K$ is returned in the `totalcost` field of the output structure. 

For example, we can access the total cost of the previously computed clustering with $K=2$ as follows:
"

# ╔═╡ 12b9ebbd-ba58-4a2f-b82f-093509d069f8
log(km_clust2.totalcost)

# ╔═╡ c043f6e3-ada2-4919-ba9b-cfa928bf6239
md"
Use this information to write a function that returns a vector of $\log W_K$ values for a given data matrix $X$ and vector of $K$ values, and then identify the kink in the plot of $\log W_K$ vs. $K$.

FYI, the complete set of fields in a `kmeans` output is:

$(fieldnames(typeof(km_clust2)))
"

# ╔═╡ a9b5eae7-e918-431b-a96c-ca5013143e18
md"
In the function below, replace the line `logW=kvec` with the correct computation of the total cost value.

**Hint:** using the `map` function, the result can be computed in a single line. Type `map` in the Live Docs to learn more about this function. If you need help seeing how, the disable the next cell, make the one below visible, and enable it.
"

# ╔═╡ b0286f6a-3409-4dc0-83eb-cecb26d7fbcf
function kmeans_totalcost(X, kvec)
	logW = kvec # Replace this with the correct computation of the total cost value!
	return logW
end

# ╔═╡ 8f5a8752-dd54-4a66-a83d-7c405e072626
# ╠═╡ disabled = true
#=╠═╡
function kmeans_totalcost(X, kvec)
	map(k -> log(kmeans(X,k).totalcost), kvec)
end
  ╠═╡ =#

# ╔═╡ 436e5378-58bc-4bc6-b71d-a899ca56e95d
kvec = 1:20

# ╔═╡ 656597ae-6773-47ea-82c4-4543fa409b6e
logW = kmeans_totalcost(X_std[:,sd.>sdcut]', kvec)

# ╔═╡ 7ecf818a-f356-4dba-82a6-e16fdae1bc06
plot(kvec, logW, xlabel=L"K", ylabel=L"\log W_K", marker=:circle,label=false)

# ╔═╡ 86c2b9b2-ca4f-49ac-9b04-a7b9d7c2fec8
md"
If the kink is hard to see (it usually is!), adding the successive differences in $\log W_K$ to the plot may help.
"

# ╔═╡ 9e150cca-2bc0-4f38-a97f-116daa46ae97
plot!(twinx(), diff(logW), ylabel=L"\log W_{K+1} - \log W_K", linecolor=:red, marker=:square, markercolor=:red,label=false)

# ╔═╡ 8a6fb2ac-d4cf-4d4d-b8c9-d3cbe1f19fd9
md"
Choose a value for the optimal $K$ based on where you identify a kink:
"

# ╔═╡ 8d032a3e-2132-42c6-9e3a-d27063777242
kopt = 5 # Change this to your chosen optimal value!

# ╔═╡ 04439ce4-85b1-4467-b851-d12ace80713e
km_opt = kmeans(X_std[:,sd.>sdcut]',kopt);

# ╔═╡ 07b2634d-d3cb-4e9e-bbd7-6758c1eab611
md"
Compute a frequency table showing the numbers of triple negative tumours in each cluster and another one showing the numbers of tumours of a given stage in each cluster. What do you observe compared to the previous results where we set $K=2$ and $K=4$?
"

# ╔═╡ 9e2e4e59-c5c4-41d6-bac4-1784edee94e9
# ╠═╡ disabled = true
#=╠═╡
freqtable(km_opt.assignments,triple_neg);
  ╠═╡ =#

# ╔═╡ 12fa001d-3b69-4994-8d3d-f58560118b65
# ╠═╡ disabled = true
#=╠═╡
freqtable(km_opt.assignments,stage);
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═6cb72390-b393-11ee-1a18-af78f69f2995
# ╠═997bb1b8-865a-4504-be9d-9aa7daa554b5
# ╠═701c1990-fffd-4d74-835d-272092d6c929
# ╠═cfab9820-de64-4d20-8354-856bf0c08a62
# ╠═39e893c4-1cde-4bbd-b464-72292513973b
# ╠═5b1a8ea9-1044-4662-8da5-6c4f025d7f84
# ╠═0d42d399-b1fa-41e6-a371-6a014d9b7161
# ╠═bc9c853d-558d-4871-a45a-b9a6db78b4ff
# ╠═0a74b8b0-af7f-4c71-a404-fb8a2b3c6404
# ╠═61466b7d-8289-4996-b956-6e231b0953f7
# ╠═f4afdd2e-77fd-499a-bf22-e52189401cc4
# ╠═cdffa0e4-db96-4822-9e59-b23d466cb42c
# ╠═bff7010b-8638-40ee-99b6-9978d77d2bdc
# ╠═86eca1aa-0606-4d69-9d83-7ad8a9189038
# ╠═10343544-6ff4-435b-96f2-6e9ca6dc9d72
# ╠═cfeb4e9f-81f5-4e7b-8d60-a049342fa104
# ╠═f940f078-ea2d-4a7f-b55b-9e09bab920d4
# ╠═3c8ac526-e0ce-45f2-b634-5aeb53f13dc9
# ╠═0ed193aa-9c4a-4f06-87b4-d81fa1252be5
# ╠═e53173a6-c32e-4ecb-997f-941803b20600
# ╠═7f14c0d2-d32d-4b34-923a-4829ce4e3322
# ╠═bece39e9-8d0d-45a6-b867-5d005062ed1a
# ╠═193ded13-8c8e-411a-b6c1-72e3b11d90fc
# ╠═2e8d67e5-a09f-48cb-a12a-49a28a84fab9
# ╠═7af16ff6-46e9-4410-99fd-8a3521f98dc3
# ╠═219bf88d-1bf7-4d27-a7c8-a71827f31876
# ╠═1d504701-65e9-443e-887d-bc77d8148b87
# ╠═4afd5aa6-34c2-41ae-b458-1d98396def5f
# ╠═f72e3abb-1ead-4e31-b69a-bab7a3292ba4
# ╠═40a718d0-510f-4e30-8b37-43f05a0d284c
# ╠═376ebe3c-89ee-4818-a04d-bfa7133252d1
# ╠═abc8da5a-9192-45c8-84d4-e61d4af5e426
# ╠═3dabdc41-e64c-4b23-a0ad-0dc8c43b8d16
# ╠═12b9ebbd-ba58-4a2f-b82f-093509d069f8
# ╠═c043f6e3-ada2-4919-ba9b-cfa928bf6239
# ╠═a9b5eae7-e918-431b-a96c-ca5013143e18
# ╠═b0286f6a-3409-4dc0-83eb-cecb26d7fbcf
# ╠═8f5a8752-dd54-4a66-a83d-7c405e072626
# ╠═436e5378-58bc-4bc6-b71d-a899ca56e95d
# ╠═656597ae-6773-47ea-82c4-4543fa409b6e
# ╠═7ecf818a-f356-4dba-82a6-e16fdae1bc06
# ╠═86c2b9b2-ca4f-49ac-9b04-a7b9d7c2fec8
# ╠═9e150cca-2bc0-4f38-a97f-116daa46ae97
# ╠═8a6fb2ac-d4cf-4d4d-b8c9-d3cbe1f19fd9
# ╠═8d032a3e-2132-42c6-9e3a-d27063777242
# ╠═04439ce4-85b1-4467-b851-d12ace80713e
# ╠═07b2634d-d3cb-4e9e-bbd7-6758c1eab611
# ╠═9e2e4e59-c5c4-41d6-bac4-1784edee94e9
# ╠═12fa001d-3b69-4994-8d3d-f58560118b65
