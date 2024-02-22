### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 964c81da-2127-492c-b813-34191a45c487
using DrWatson

# ╔═╡ 10e0e352-a516-4a92-b5ab-48bd81e06d0c
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ 7206ec51-a609-4ca4-b6dc-92b4781bbed1
begin
	using DataFrames
	using Arrow
	using CSV
	using DataSets
	using PlutoUI
end

# ╔═╡ 4e1b5250-cf2f-11ee-2c28-372723b57034
md"# Dimensionality reduction for single-cell RNA-seq data

## Setup the environment

Activate the BINF301 environment and load packages:
"

# ╔═╡ 7c36ca21-eb71-4839-8dde-57efac1cac5a
md"""
## Load the data

Make sure that you have downloaded the data before running this notebook by executing the script `download_processed_data.jl` in the `scripts` folder. Information to load the data is stored in the `Data.toml` file, which needs to be loaded first:
"""

# ╔═╡ f458c335-0ba3-4ed6-bf2d-7d847667ce26
DataSets.load_project!(projectdir("Data.toml"));

# ╔═╡ a55398f6-f9cc-4ba7-adf0-0f1c85405caa
tree = DataSets.open(dataset("Mouse_V1_ALM"));

# ╔═╡ f324e478-11d4-4b9e-a839-0d73d30042ba
md"Read the single-cell expression data. These data have been prefiltered by discarding all genes with non-zero expression (more than 32 counts) in less that 10 cells, following [Kobak & Berens (2019)](https://doi.org/10.1038/s41467-019-13056-x)."

# ╔═╡ c6fabbae-6e90-458c-b121-8615ead04f8a
begin
	io = open(IO, tree["mouse_ALM_VISp_gene_expression.arrow"])
	stream = Arrow.Stream(io);
	df_expr = [DataFrame(table) for table in stream][1]
	close(io)
end

# ╔═╡ ff586f69-64d6-42e7-9496-3960301f7e8e
md"
Columns in the expression data correspond to cells and rows to genes. We will not use any gene annotation in this notebook, but it can be loaded as follows:
"

# ╔═╡ 3ce46687-f16f-4bdc-a0ad-3c386b2c16d7
df_annot = open(Vector{UInt8}, tree["mouse_ALM_VISp_gene_annotation.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 6c3fc69c-c29f-44dc-86f9-067ecbd2f990
md"Read the cluster labels from [Tasic et al. (2018)](https://doi.org/10.1038/s41586-018-0654-5):"

# ╔═╡ dfdd9693-e3ef-4371-a7de-c7251c04df35
df_clust = open(Vector{UInt8}, tree["tasic-sample_heatmap_plot_data.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 30c84cf7-2ead-417e-8183-e11530f7a5c9
md"""
## Kobak and Berens pipeline

### Sequencing depth normalization

We start by computing the library depth per million for each cell as it will be needed later.
"""

# ╔═╡ 457a67ef-22d8-4e7b-b3ff-50bf071f7850
libraryDepth = sum.(eachcol(df_expr)) / 1e6 ;

# ╔═╡ 7437fb10-9a0a-4ef7-89c4-b939f783255f
md"""
### Feature selection

We say a gene has non-zero expression in a cell if the count is at least ``t=`` $(32): 
"""

# ╔═╡ 414accfb-861d-4361-860c-6ab8b50a611e
t = 32;

# ╔═╡ dbe18d66-bca8-4e3a-a63e-f67b1eaabb0a
expr_is_nonzero = df_expr .> t;

# ╔═╡ e1c71a60-696e-4523-a5d3-de7cd5c431c2
md"Compute the fraction of cells with near-zero (less than ``t=``$(t)) counts for each gene."

# ╔═╡ e515ddac-e965-4ff6-84be-1b9c4c95a309
ncell = ncol(df_expr);

# ╔═╡ 8f2e17e2-7ff3-47db-be59-665fc18a66ac
d = select(expr_is_nonzero, AsTable(:) => ByRow(x -> sum(x)) => :d);

# ╔═╡ abc935d2-beb1-42c7-88b0-03e76be3e577
select!(d, :d => (x -> x/ncell), renamecols=false)

# ╔═╡ b1b2b0c0-c415-476b-b550-1b637bf961d4
md"Compute the mean log non-zero expression for each gene:"

# ╔═╡ b8124b1d-db1f-445c-b34f-89d1eec5a00d
# ╠═╡ disabled = true
#=╠═╡
begin
	m = zeros(nrow(df_expr))
	for k = eachindex(m)
		m[k] = mean(log2.(df_expr[k, expr_is_nonzero[k,:]]))
	end
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─4e1b5250-cf2f-11ee-2c28-372723b57034
# ╠═964c81da-2127-492c-b813-34191a45c487
# ╠═10e0e352-a516-4a92-b5ab-48bd81e06d0c
# ╠═7206ec51-a609-4ca4-b6dc-92b4781bbed1
# ╟─7c36ca21-eb71-4839-8dde-57efac1cac5a
# ╠═f458c335-0ba3-4ed6-bf2d-7d847667ce26
# ╠═a55398f6-f9cc-4ba7-adf0-0f1c85405caa
# ╟─f324e478-11d4-4b9e-a839-0d73d30042ba
# ╠═c6fabbae-6e90-458c-b121-8615ead04f8a
# ╟─ff586f69-64d6-42e7-9496-3960301f7e8e
# ╠═3ce46687-f16f-4bdc-a0ad-3c386b2c16d7
# ╟─6c3fc69c-c29f-44dc-86f9-067ecbd2f990
# ╠═dfdd9693-e3ef-4371-a7de-c7251c04df35
# ╟─30c84cf7-2ead-417e-8183-e11530f7a5c9
# ╠═457a67ef-22d8-4e7b-b3ff-50bf071f7850
# ╟─7437fb10-9a0a-4ef7-89c4-b939f783255f
# ╠═414accfb-861d-4361-860c-6ab8b50a611e
# ╠═dbe18d66-bca8-4e3a-a63e-f67b1eaabb0a
# ╟─e1c71a60-696e-4523-a5d3-de7cd5c431c2
# ╠═e515ddac-e965-4ff6-84be-1b9c4c95a309
# ╠═8f2e17e2-7ff3-47db-be59-665fc18a66ac
# ╠═abc935d2-beb1-42c7-88b0-03e76be3e577
# ╟─b1b2b0c0-c415-476b-b550-1b637bf961d4
# ╠═b8124b1d-db1f-445c-b34f-89d1eec5a00d
