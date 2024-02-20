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
md"Read the expression data:"

# ╔═╡ c6fabbae-6e90-458c-b121-8615ead04f8a
begin
	io = open(IO, tree["mouse_ALM_VISp_gene_expression.arrow"])
	stream = Arrow.Stream(io);
	df_expr = [DataFrame(table) for table in stream][1]
	close(io)
end

# ╔═╡ 12cc6875-054c-4d46-bb98-cf2b18351678
md"We will not need gene names in this notebook:"

# ╔═╡ 84d53176-a516-441c-b1dc-be0562470ece
select!(df_expr, Not(1));

# ╔═╡ 754adb91-24bb-4887-a97e-cc476e25544a
X = Matrix(df_expr[!,2:end])

# ╔═╡ 6c3fc69c-c29f-44dc-86f9-067ecbd2f990
md"Read the cluster labels:"

# ╔═╡ dfdd9693-e3ef-4371-a7de-c7251c04df35
df_clust = open(Vector{UInt8}, tree["tasic-sample_heatmap_plot_data.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 8fe41526-f3aa-434b-9598-077a0ed409fb
md"Move the slider to change the value of ``t`` (default 32):"

# ╔═╡ 71c93d56-6c7d-4dc4-8d46-04c0179e8b0b
@bind t Slider(10:50; default=32)

# ╔═╡ 0d55e6ef-3919-4c59-b4a5-d61b2540e52f
md"Move the slider to change the value of ``n_{\min}`` (default 10):"

# ╔═╡ 8f75e5c1-f53b-437a-8712-40411f429494
@bind nmin Slider(1:30; default=10)

# ╔═╡ 30c84cf7-2ead-417e-8183-e11530f7a5c9
md"""
## Kobak and Berens pipeline

### Feature selection

Genes that have non-zero expression (that is, expression greater than a threshold ``t=`` $(t) in less than ``n_{\min}=`` $(nmin) cells are discarded.
"""

# ╔═╡ 2343a538-468f-4d06-ad9d-6951d2e98ba7
md"We start by computing the library depth per million for each cell as it will be needed later."

# ╔═╡ 457a67ef-22d8-4e7b-b3ff-50bf071f7850
libraryDepth = map( x -> sum(x)/1e6, eachcol(X));

# ╔═╡ 4688f4fc-6c8b-46e4-a34b-8fd4f387a87f
sum(X,dims=1)

# ╔═╡ f78b9f0b-39ea-4632-b2dc-8b41699935f8
tf = sum(eachcol(df_expr .>= t)) .>= nmin;

# ╔═╡ d2279d6e-2306-4e18-b5f4-9732763040ee
size(tf)

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
# ╟─12cc6875-054c-4d46-bb98-cf2b18351678
# ╠═84d53176-a516-441c-b1dc-be0562470ece
# ╠═754adb91-24bb-4887-a97e-cc476e25544a
# ╟─6c3fc69c-c29f-44dc-86f9-067ecbd2f990
# ╠═dfdd9693-e3ef-4371-a7de-c7251c04df35
# ╟─30c84cf7-2ead-417e-8183-e11530f7a5c9
# ╟─8fe41526-f3aa-434b-9598-077a0ed409fb
# ╟─71c93d56-6c7d-4dc4-8d46-04c0179e8b0b
# ╟─0d55e6ef-3919-4c59-b4a5-d61b2540e52f
# ╟─8f75e5c1-f53b-437a-8712-40411f429494
# ╟─2343a538-468f-4d06-ad9d-6951d2e98ba7
# ╠═457a67ef-22d8-4e7b-b3ff-50bf071f7850
# ╠═4688f4fc-6c8b-46e4-a34b-8fd4f387a87f
# ╠═f78b9f0b-39ea-4632-b2dc-8b41699935f8
# ╠═d2279d6e-2306-4e18-b5f4-9732763040ee
