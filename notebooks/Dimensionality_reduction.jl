### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ d37cdfa8-a145-480d-83c9-57ae74ce99f0
using DrWatson

# ╔═╡ 69afbe07-3966-4fd5-8660-c2ca82c9c37c
quickactivate(@__DIR__)

# ╔═╡ 4ac756a5-5bfa-4c7a-a471-ebbc5b93922f
begin
	using DataFrames
	using CSV
	using PlutoUI
	using Statistics
	using SingleCellProjections
	using SparseArrays
	#using JLD
	using StatsPlots
	using LaTeXStrings
	using TSne
	using UMAP
end

# ╔═╡ 9bf0196e-29ca-11f1-8285-3551efeea6f4
md"# Dimensionality reduction for single-cell RNA-seq data

## Setup the environment

Activate the BINF301 environment and load packages:
"

# ╔═╡ 85ebdee4-f4d0-484f-9560-b22e20dfa81f
md"""
## Load the data

### Single-cell RNA-seq data

Read the single-cell expression data. These data have been prefiltered by discarding all genes with non-zero expression (more than 32 counts) in less that 10 cells, following [Kobak & Berens (2019)](https://doi.org/10.1038/s41467-019-13056-x). The original data is available on [Brain-map](https://brain-map.org/our-research/cell-types-taxonomies/cell-types-database-rna-seq-data/mouse-v1-and-alm-smart-seq). Columns correspond to cells and rows to genes.
"""

# ╔═╡ 75dccfdd-b33f-4f02-9c2c-227fe1d206f9
fexpr = datadir("processed", "Mouse_V1_ALM_subset", "mouse_ALM_VISp_gene_expression.csv");

# ╔═╡ 8e29df01-0b05-42f0-85b9-1ee75ba10dee
df_expr = DataFrame(CSV.File(fexpr))

# ╔═╡ 938f8027-9e92-4fae-a1b4-d3ba8e47f3fb
md"For some operations it's easier to work with a matrix:"

# ╔═╡ c467acbf-8c31-4a90-80bb-a802eb4a7038
counts = Matrix(df_expr)

# ╔═╡ f2f7263d-0123-4b61-b55d-dfa9ebb3050c
md"""
### Gene and cell annotation data

The following two tables give information about the genes and provide a set of cell cluster labels from [Tasic et al. (2018)](https://doi.org/10.1038/s41586-018-0654-5):
"""

# ╔═╡ eede3bf1-037a-4eb2-8885-f3e52d9b5acd
fgene_annot = datadir("processed", "Mouse_V1_ALM_subset", "mouse_ALM_VISp_gene_annotation.csv");

# ╔═╡ 13ff8304-4643-4202-aaa0-4ebf335ec462
df_gene_annot = DataFrame(CSV.File(fgene_annot))

# ╔═╡ a5aa669a-ea63-4bf3-8c8c-0670a277b691
fcell_annot = datadir("processed", "Mouse_V1_ALM_subset", "tasic-sample_heatmap_plot_data.csv");

# ╔═╡ 8c803d88-a89f-4e70-89b7-beaf120f9511
df_cell_annot = DataFrame(CSV.File(fcell_annot))

# ╔═╡ 733b5f9c-bd5a-4cec-81af-c727d8204ec4
md"""
## Kobak and Berens pipeline

### Sequencing depth normalization

We start by computing the library depth per million for each cell as it will be needed later.
"""

# ╔═╡ ef900545-e930-4c2c-9490-a2a3d510225e
libraryDepth = [sum(col)/1e6 for col in eachcol(df_expr)];

# ╔═╡ ecede165-f1aa-4309-89fa-ec7cfc8cb9a4
histogram(libraryDepth, xlabel="Library depth [million reads]", label="")

# ╔═╡ bd14fb35-5bae-4314-99e6-2a8ac851dede
md"""
### Feature selection

We say a gene has non-zero expression in a cell if the count is at least ``t=32``.  The preprocessing script for our data has already removed all count values less than ``t`` and all genes that have non-zero expression in less than 10 cells. We define the threshold value for consistency.
"""

# ╔═╡ c2939eef-ff71-4f76-b0df-c7aff8668a09
t = 32;

# ╔═╡ cf78f9fa-1ac1-4122-ad28-ab28fb7d7c57
md" First we compute the number of cells with non-zero counts for each gene:"

# ╔═╡ 2e9b20f8-fc3d-4230-bbe5-ae975f3ceec8
n = vec(sum(counts .>= t, dims=2));

# ╔═╡ bb4ca372-c923-4b38-a116-f6dcb08753ee
md"The fraction of cells with non-zero counts for each gene is computed as:"

# ╔═╡ 4424b320-b5b9-4d0e-a5f4-25905e02354c
ncell = ncol(df_expr);

# ╔═╡ b67ae755-5589-4429-95fc-487ca3017a1f
dg = 1 .- n./ncell;

# ╔═╡ 59b55605-1813-4e95-b047-29cb1290312f
md"Now compute the sum of log2-counts over all cells with non-zero counts for each gene, using a function that returns ``log_2(x)`` if an expression count ``x`` is greater than ``t`` and zero otherwise, and divide the result by ``n`` elementwise to obtain the mean log non-zero expression for each gene:"

# ╔═╡ 06759942-1f49-464d-b88f-1308afa055f1
g(x) = x .>= t ? log2(x) : 0.

# ╔═╡ df878c40-8491-4732-be40-c735ff679d6b
mg =  vec(sum(g.(counts), dims=2)) ./ n;

# ╔═╡ 98580232-08c5-4498-8176-a324d0d00a00
md"Use the same formula as Kobak & Berens to select genes, with parameters from their Supp Fig 4."

# ╔═╡ d73f1669-2f9a-49c8-bbee-fa39520e207f
begin
	a = 1.5;
	b = 6.56;
	featureSelect = (dg .> exp.(-a.*(mg.-b)) .+ 0.02);
end;

# ╔═╡ d79db859-9e4a-472b-8bb5-6236c0199e32
begin
	x = range(minimum(mg),maximum(mg),length=100);
	scatter(mg, dg,
		label = "",
		xlabel = "Mean log2 nonzero expression",
		ylabel = "Frequency of nonzero expression"
	)
	plot!(x, exp.(-a.*(x.-b)) .+ 0.02, color=:red, linewidth=3, label="")
	ylims!(0, 1.0)
	annotate!(12.5,0.2,L"y=\exp(-1.5(x-6.56))+0.02", :red)
end

# ╔═╡ 54ad9c21-b1ff-48ac-8cfd-274f3f7cbc48
md"""
### Non-linear transformation

Filter the original count data with the selected features, normalize cells by their library depth, transform all values with a ``\log2(x+1)`` transformation and store the result a [DataMatrix object](https://biojulia.dev/SingleCellProjections.jl/dev/datamatrices/) with row and column annotations, such that we can use the [SingleCellProjections](https://github.com/BioJulia/SingleCellProjections.jl) package for the rest of the analysis.
"""

# ╔═╡ 03f3cd30-f6f2-4d67-8704-1a105c124302
dm = DataMatrix(log2.( counts[featureSelect,:] ./ libraryDepth' .+ 1.), df_gene_annot[featureSelect,:], df_cell_annot)

# ╔═╡ a3f70fd0-6136-483c-9318-607b95e985b8
npc = 50;

# ╔═╡ 13c67ef4-f3f7-43d9-a0ad-87da4b2464ba
md"""
### Principal component analysis

Reduce the size of the data to $(npc) dimensions prior to running t-SNE.
"""

# ╔═╡ 7638c17a-58cd-4e52-adf7-c8192bd00ff8
dm_reduced = svd(dm; nsv=npc)

# ╔═╡ 8a19c2a1-eda4-4cc2-8e6e-c1307c4fd985
md"
### t-SNE

Because t-SNE doesn't scale well with the number of cells, we run it on every 5th cell
"

# ╔═╡ 62e925b0-fa8d-4ba8-9dad-f8ab26ce2f53
subc = 1:5:ncell

# ╔═╡ d3f54647-4a42-42dc-9c13-027f48c3e0aa
dm_t = tsne(obs_coordinates(dm_reduced)[:,subc]', 2)

# ╔═╡ baf86246-de14-4994-96bb-b0620ce828fe
md"""### UMAP

UMAP does not have the same scaling issues as t-SNE and runs fine on the whole dataset, but we run it on the subset for simplicity:
"""

# ╔═╡ 59fd06e7-83f1-49f4-b2dc-e46afee27e8a
dm_u = umap(obs_coordinates(dm_reduced)[:,subc], 2)

# ╔═╡ 9d3fb6c0-efe5-43c6-850d-de82ac8da1e4
md"
## Figures

Use 1 in 5 cells to ease computation and color the cells using the cell annotation.
"

# ╔═╡ 46550f2c-2ef1-4009-bace-a1a5bb011e45
begin
	clid = dm_reduced.obs.cluster_id[subc];
	clcol = dm_reduced.obs.cluster_color[subc];
	uu = unique(clcol);
end;

# ╔═╡ 1c5ea6b9-1500-45c4-b179-d5e272b87aad
md"
### PCA

Plot the first 2 PCs against each other:
"

# ╔═╡ f7524305-fd67-4087-bc2a-ecaadee9b1c3
pc1 = obs_coordinates(dm_reduced)[1,subc];

# ╔═╡ f1b922a0-d459-4f80-828e-3bda5dd38173
pc2 = obs_coordinates(dm_reduced)[2,subc];

# ╔═╡ af53d7f0-3d66-46fa-b8ac-e139a38946c7
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

# ╔═╡ 91e42563-c431-4b48-8382-21859eb36a95
md"
### t-SNE

Plot the two t-SNE coordinates against each other:
"

# ╔═╡ 2afec4cf-7d68-471d-b3bc-5ec58ee5f73a
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

# ╔═╡ efb27277-87ab-4ca7-ab3b-d8f9ddfbd1be
md"
### UMAP

Plot the two UMAP coordinates against each other:
"

# ╔═╡ c8646566-037e-4aa5-b6c7-ff0d0ab2fd8e
um1 = dm_u[1,:];

# ╔═╡ 9c98fa02-356f-478b-bfda-9dd7f75458fd
um2 = dm_u[2,:];

# ╔═╡ a4140680-e618-4c7d-9d21-ac149df240a3
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
# ╟─9bf0196e-29ca-11f1-8285-3551efeea6f4
# ╠═d37cdfa8-a145-480d-83c9-57ae74ce99f0
# ╠═69afbe07-3966-4fd5-8660-c2ca82c9c37c
# ╠═4ac756a5-5bfa-4c7a-a471-ebbc5b93922f
# ╟─85ebdee4-f4d0-484f-9560-b22e20dfa81f
# ╠═75dccfdd-b33f-4f02-9c2c-227fe1d206f9
# ╠═8e29df01-0b05-42f0-85b9-1ee75ba10dee
# ╟─938f8027-9e92-4fae-a1b4-d3ba8e47f3fb
# ╠═c467acbf-8c31-4a90-80bb-a802eb4a7038
# ╠═f2f7263d-0123-4b61-b55d-dfa9ebb3050c
# ╠═eede3bf1-037a-4eb2-8885-f3e52d9b5acd
# ╠═13ff8304-4643-4202-aaa0-4ebf335ec462
# ╠═a5aa669a-ea63-4bf3-8c8c-0670a277b691
# ╠═8c803d88-a89f-4e70-89b7-beaf120f9511
# ╟─733b5f9c-bd5a-4cec-81af-c727d8204ec4
# ╠═ef900545-e930-4c2c-9490-a2a3d510225e
# ╠═ecede165-f1aa-4309-89fa-ec7cfc8cb9a4
# ╟─bd14fb35-5bae-4314-99e6-2a8ac851dede
# ╠═c2939eef-ff71-4f76-b0df-c7aff8668a09
# ╟─cf78f9fa-1ac1-4122-ad28-ab28fb7d7c57
# ╠═2e9b20f8-fc3d-4230-bbe5-ae975f3ceec8
# ╟─bb4ca372-c923-4b38-a116-f6dcb08753ee
# ╠═4424b320-b5b9-4d0e-a5f4-25905e02354c
# ╠═b67ae755-5589-4429-95fc-487ca3017a1f
# ╟─59b55605-1813-4e95-b047-29cb1290312f
# ╠═06759942-1f49-464d-b88f-1308afa055f1
# ╠═df878c40-8491-4732-be40-c735ff679d6b
# ╟─98580232-08c5-4498-8176-a324d0d00a00
# ╠═d73f1669-2f9a-49c8-bbee-fa39520e207f
# ╠═d79db859-9e4a-472b-8bb5-6236c0199e32
# ╟─54ad9c21-b1ff-48ac-8cfd-274f3f7cbc48
# ╠═03f3cd30-f6f2-4d67-8704-1a105c124302
# ╟─13c67ef4-f3f7-43d9-a0ad-87da4b2464ba
# ╠═a3f70fd0-6136-483c-9318-607b95e985b8
# ╠═7638c17a-58cd-4e52-adf7-c8192bd00ff8
# ╟─8a19c2a1-eda4-4cc2-8e6e-c1307c4fd985
# ╠═62e925b0-fa8d-4ba8-9dad-f8ab26ce2f53
# ╠═d3f54647-4a42-42dc-9c13-027f48c3e0aa
# ╟─baf86246-de14-4994-96bb-b0620ce828fe
# ╠═59fd06e7-83f1-49f4-b2dc-e46afee27e8a
# ╟─9d3fb6c0-efe5-43c6-850d-de82ac8da1e4
# ╠═46550f2c-2ef1-4009-bace-a1a5bb011e45
# ╟─1c5ea6b9-1500-45c4-b179-d5e272b87aad
# ╠═f7524305-fd67-4087-bc2a-ecaadee9b1c3
# ╠═f1b922a0-d459-4f80-828e-3bda5dd38173
# ╠═af53d7f0-3d66-46fa-b8ac-e139a38946c7
# ╟─91e42563-c431-4b48-8382-21859eb36a95
# ╠═2afec4cf-7d68-471d-b3bc-5ec58ee5f73a
# ╟─efb27277-87ab-4ca7-ab3b-d8f9ddfbd1be
# ╠═c8646566-037e-4aa5-b6c7-ff0d0ab2fd8e
# ╠═9c98fa02-356f-478b-bfda-9dd7f75458fd
# ╠═a4140680-e618-4c7d-9d21-ac149df240a3
