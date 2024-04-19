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
	using Statistics
	using SingleCellProjections
	using SparseArrays
	using JLD
	using StatsPlots
	using LaTeXStrings
	using TSne
	using UMAP
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
tree = DataSets.open(dataset("Mouse_V1_ALM"))

# ╔═╡ f324e478-11d4-4b9e-a839-0d73d30042ba
md"Read the single-cell expression data. These data have been prefiltered by discarding all genes with non-zero expression (more than 32 counts) in less that 10 cells, following [Kobak & Berens (2019)](https://doi.org/10.1038/s41467-019-13056-x). The columns in the expression dataframe are the row indices, column indices, and values to create a sparse count matrix where cells are columns and rows are genes. We read the dataframe, do the conversion to a sparse matrix and then discard the dataframe:"

# ╔═╡ c6fabbae-6e90-458c-b121-8615ead04f8a
begin
	io = open(IO, tree["mouse_ALM_VISp_gene_expression.arrow"])
	stream = Arrow.Stream(io);
	df_expr = [DataFrame(table) for table in stream][1]
	close(io)
	counts = sparse(df_expr.I, df_expr.J, df_expr.V);
	df_expr = nothing
end

# ╔═╡ 5524ba5a-15b0-4f9b-b33e-e8ada27eaeef
nnz(counts)  / prod(size(counts))

# ╔═╡ 4d5b6a30-9b89-494b-b4ec-b7c80078df27
md"
Read the gene annotation;
"

# ╔═╡ 3ce46687-f16f-4bdc-a0ad-3c386b2c16d7
df_gene_annot = open(Vector{UInt8}, tree["mouse_ALM_VISp_gene_annotation.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 6c3fc69c-c29f-44dc-86f9-067ecbd2f990
md"Read the cluster labels from [Tasic et al. (2018)](https://doi.org/10.1038/s41586-018-0654-5):"

# ╔═╡ dfdd9693-e3ef-4371-a7de-c7251c04df35
df_cell_annot = open(Vector{UInt8}, tree["tasic-sample_heatmap_plot_data.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 30c84cf7-2ead-417e-8183-e11530f7a5c9
md"""
## Kobak and Berens pipeline

### Sequencing depth normalization

We start by computing the library depth per million for each cell as it will be needed later.
"""

# ╔═╡ 457a67ef-22d8-4e7b-b3ff-50bf071f7850
libraryDepth = sum(counts, dims=1) / 1e6;

# ╔═╡ 8fa30134-6b07-4bb5-a155-a7b70f40218a
histogram(libraryDepth)

# ╔═╡ b29d6140-c74f-47f7-82fb-36e1691f2d66


# ╔═╡ 7437fb10-9a0a-4ef7-89c4-b939f783255f
md"""
### Feature selection

We say a gene has non-zero expression in a cell if the count is at least ``t=32``.  The preprocessing script for our data has already removed all count values less than ``t`` and all genes that have non-zero expression in less than 10 cells. Hence we can use zero as the threshold in our calculations, which is fast due to the `counts` matrix being represented as a a sparse matrix.
"""

# ╔═╡ c68e367d-8788-44f5-af03-af14e533ad8f
md"The total number of cells in our data is:"

# ╔═╡ 4112422d-7472-40ea-8a1e-6a2cb66d1776
ncell = size(counts,2)

# ╔═╡ e1c71a60-696e-4523-a5d3-de7cd5c431c2
md"Compute the number of cells with non-zero counts for each gene:"

# ╔═╡ d695a1a3-7109-40c6-99d1-21924023a682
n = vec(sum(counts .> 0, dims=2));

# ╔═╡ 48b67b53-4aaf-4f38-b09f-598d98cc549b
md"The fraction of cells with non-zero counts for each gene is computed as:"

# ╔═╡ d1506893-fe0a-4471-9a8c-03cf2e345a85
d = 1 .- n./ncell;

# ╔═╡ 781d2b01-5d51-46a5-835e-6d023f892993
md"Now compute the sum of log2-counts over all cells with non-zero counts for each gene, using a function that returns ``log_2(x)`` if an expression count ``x`` is greater than zero and zero otherwise, and divide the result by ``n`` elementwise to obtain the mean log non-zero expression for each gene:"

# ╔═╡ 9b4a653f-faa0-41ff-bb7e-b12dba5a7627
g(x) = x .> 0 ? log2(x) : 0.

# ╔═╡ 4dbd070e-4ff1-47e7-945c-f09616a2f982
m =  vec(sum(g.(counts), dims=2)) ./ n

# ╔═╡ 1d2700c2-3f62-400f-9c8e-32732909e7f8
md"Use the same formula as Kobak & Berens to select genes, with parameters from their Supp Fig 4."

# ╔═╡ 11b6a951-75f1-4612-9ab8-3b919590c642
begin
	a = 1.5;
	b = 6.56;
	featureSelect = d .> exp.(-a.*(m.-b)) .+ 0.02;
end

# ╔═╡ 72218b73-f4a5-4bac-8f30-d76206ee2ee7
begin
	x = range(minimum(m),maximum(m),length=100);
	scatter(m,d,
		label = "",
		xlabel = "Mean log2 nonzero expression",
		ylabel = "Frequency of nonzero expression"
	)
	plot!(x, exp.(-a.*(x.-b)) .+ 0.02, color=:red, linewidth=3, label="")
	ylims!(0, 1.0)
	annotate!(12.5,0.2,L"y=\exp(-1.5(x-6.56))+0.02", :red)
end

# ╔═╡ f2f23bec-e2d0-45ed-9175-82faaf5d7d69
md"""
### Non-linear transformation

Filter the original count data with the selected features, normalize cells by their library depth, transform all values with a ``\log2(x+1)`` transformation and store the result a [DataMatrix object](https://biojulia.dev/SingleCellProjections.jl/dev/datamatrices/) with row and column annotations, such that we can use the [SingleCellProjections](https://github.com/BioJulia/SingleCellProjections.jl) package for the rest of the analysis.
"""

# ╔═╡ 393e4803-a386-4e1d-8c92-2033dbe70f4a
dm = DataMatrix(log2.( counts[featureSelect,:] ./ libraryDepth .+ 1.), df_gene_annot[featureSelect,:], df_cell_annot)

# ╔═╡ 02d42ae4-2228-4de9-8e74-6720195850e9
npc = 50;

# ╔═╡ 7aba00e8-ce1c-47e5-a8da-7253dd17f335
md"""
### Principal component analysis

Reduce the size of the data to $(npc) dimensions prior to running t-SNE.
"""

# ╔═╡ 5d5ce339-449e-49fb-a5a1-1937050a3927
dm_reduced = svd(dm; nsv=npc)

# ╔═╡ bae16d51-daed-40b2-85b3-94ba658f2acb
md"
### t-SNE
"

# ╔═╡ 0abca9d4-2cbf-4d02-9812-8d389d69dc87
dm_u = umap(obs_coordinates(dm_reduced), 2)

# ╔═╡ 3a9c2491-fce4-436d-b335-456c504bebb5
md"
## Figures

Use 1 in 5 cells to ease computation and color the cells using the cell annotation.
"

# ╔═╡ edbcbf6a-9f4c-4059-93c9-be1b89c91fb5
subc = 1:5:ncell;

# ╔═╡ 1e99e15d-5199-4cb1-9092-38c02a9af49c
# ╠═╡ show_logs = false
dm_t = tsne(obs_coordinates(dm_reduced)[:,subc]', 2)

# ╔═╡ 87e4ef6d-206e-4e1c-844c-5f93da0227eb
begin
	clid = dm_reduced.obs.cluster_id[subc];
	clcol = dm_reduced.obs.cluster_color[subc];
	uu = unique(clcol);
end

# ╔═╡ e4b90eb3-f1e5-42f8-81ba-357d28b702e1
md"
### PCA

Plot the first 2 PCs against each other:
"

# ╔═╡ 4a4ea89a-ba7f-43f2-ab72-14bd2b062e48
pc1 = obs_coordinates(dm_reduced)[1,subc];

# ╔═╡ 1a63a434-f623-46f9-aa7b-d7ea0fe47a39
pc2 = obs_coordinates(dm_reduced)[2,subc];

# ╔═╡ c066bbd7-3c77-4e59-af80-88e561d74185
begin
	f1 = scatter(pc1,pc2,
		label="",
		xlabel="PCA 1",
		ylabel="PCA 2",
		markerstrokewidth = 0.5
	)
	for col in uu
	    sel = isequal.(clcol,col)
	    scatter!(pc1[sel],pc2[sel],
			color=parse(Colorant, col),
			label="",
			markerstrokewidth = 0
		)
	end
	f1
end

# ╔═╡ 3278f614-3009-4cb3-be32-2228e3cc1372
md"
### T-SNE

Plot the two t-SNE coordinates against each other:
"

# ╔═╡ 4eae61e7-b578-4345-8cf7-d0c05a1a5ff2
begin
	f2 = scatter(dm_t[:,1],dm_t[:,2],
		label="",
		xlabel="tSNE 1",
		ylabel="tSNE 2",
		markerstrokewidth = 0.5
	)
	for col in uu
	    sel = isequal.(clcol,col)
	    scatter!(dm_t[sel,1],dm_t[sel,2],
			color=parse(Colorant, col),
			label="",
			markerstrokewidth = 0
		)
	end
	f2
end

# ╔═╡ c6485ac6-665c-4acb-ac45-d3483ca18f94
md"
### UMAP

Plot the two UMAP coordinates against each other:
"

# ╔═╡ 14df6889-48ba-4ed2-aafe-94ee857f6bea
um1 = dm_u[1,subc];

# ╔═╡ 48e914dd-f898-46a2-9279-36b365bdef35
um2 = dm_u[2,subc];

# ╔═╡ 14a87a93-c26f-4e24-b294-27be92708d8d
begin
	f3 = scatter(um1,um2,
		label="",
		xlabel="UMAP 1",
		ylabel="UMAP 2",
		markerstrokewidth = 0.5
	)
	for col in uu
	    sel = isequal.(clcol,col)
	    scatter!(um1[sel],um2[sel],
			color=parse(Colorant, col),
			label="",
			markerstrokewidth = 0
		)
	end
	f3
end

# ╔═╡ Cell order:
# ╠═4e1b5250-cf2f-11ee-2c28-372723b57034
# ╠═964c81da-2127-492c-b813-34191a45c487
# ╠═10e0e352-a516-4a92-b5ab-48bd81e06d0c
# ╠═7206ec51-a609-4ca4-b6dc-92b4781bbed1
# ╠═7c36ca21-eb71-4839-8dde-57efac1cac5a
# ╠═f458c335-0ba3-4ed6-bf2d-7d847667ce26
# ╠═a55398f6-f9cc-4ba7-adf0-0f1c85405caa
# ╟─f324e478-11d4-4b9e-a839-0d73d30042ba
# ╠═c6fabbae-6e90-458c-b121-8615ead04f8a
# ╠═5524ba5a-15b0-4f9b-b33e-e8ada27eaeef
# ╟─4d5b6a30-9b89-494b-b4ec-b7c80078df27
# ╠═3ce46687-f16f-4bdc-a0ad-3c386b2c16d7
# ╟─6c3fc69c-c29f-44dc-86f9-067ecbd2f990
# ╠═dfdd9693-e3ef-4371-a7de-c7251c04df35
# ╟─30c84cf7-2ead-417e-8183-e11530f7a5c9
# ╠═457a67ef-22d8-4e7b-b3ff-50bf071f7850
# ╠═8fa30134-6b07-4bb5-a155-a7b70f40218a
# ╠═b29d6140-c74f-47f7-82fb-36e1691f2d66
# ╟─7437fb10-9a0a-4ef7-89c4-b939f783255f
# ╟─c68e367d-8788-44f5-af03-af14e533ad8f
# ╠═4112422d-7472-40ea-8a1e-6a2cb66d1776
# ╟─e1c71a60-696e-4523-a5d3-de7cd5c431c2
# ╠═d695a1a3-7109-40c6-99d1-21924023a682
# ╟─48b67b53-4aaf-4f38-b09f-598d98cc549b
# ╠═d1506893-fe0a-4471-9a8c-03cf2e345a85
# ╟─781d2b01-5d51-46a5-835e-6d023f892993
# ╠═9b4a653f-faa0-41ff-bb7e-b12dba5a7627
# ╠═4dbd070e-4ff1-47e7-945c-f09616a2f982
# ╟─1d2700c2-3f62-400f-9c8e-32732909e7f8
# ╠═11b6a951-75f1-4612-9ab8-3b919590c642
# ╠═72218b73-f4a5-4bac-8f30-d76206ee2ee7
# ╟─f2f23bec-e2d0-45ed-9175-82faaf5d7d69
# ╠═393e4803-a386-4e1d-8c92-2033dbe70f4a
# ╟─7aba00e8-ce1c-47e5-a8da-7253dd17f335
# ╠═02d42ae4-2228-4de9-8e74-6720195850e9
# ╠═5d5ce339-449e-49fb-a5a1-1937050a3927
# ╠═bae16d51-daed-40b2-85b3-94ba658f2acb
# ╠═1e99e15d-5199-4cb1-9092-38c02a9af49c
# ╠═0abca9d4-2cbf-4d02-9812-8d389d69dc87
# ╟─3a9c2491-fce4-436d-b335-456c504bebb5
# ╠═edbcbf6a-9f4c-4059-93c9-be1b89c91fb5
# ╠═87e4ef6d-206e-4e1c-844c-5f93da0227eb
# ╟─e4b90eb3-f1e5-42f8-81ba-357d28b702e1
# ╠═4a4ea89a-ba7f-43f2-ab72-14bd2b062e48
# ╠═1a63a434-f623-46f9-aa7b-d7ea0fe47a39
# ╠═c066bbd7-3c77-4e59-af80-88e561d74185
# ╟─3278f614-3009-4cb3-be32-2228e3cc1372
# ╠═4eae61e7-b578-4345-8cf7-d0c05a1a5ff2
# ╟─c6485ac6-665c-4acb-ac45-d3483ca18f94
# ╠═14df6889-48ba-4ed2-aafe-94ee857f6bea
# ╠═48e914dd-f898-46a2-9279-36b365bdef35
# ╠═14a87a93-c26f-4e24-b294-27be92708d8d
