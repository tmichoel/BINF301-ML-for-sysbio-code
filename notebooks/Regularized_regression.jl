### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ ada02ef3-ca52-4b63-a138-fa084904fd39
using DrWatson

# ╔═╡ 57fffbda-1158-4f8c-9e58-46f19ac1276d
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ 0aae0966-c1e1-4e7c-bfd7-711778cfb127
begin
	using DataFrames
	using CSV
	using DataSets
	using MLJ
	using MLJLinearModels
	using Statistics
	using StatsPlots, LaTeXStrings
	using PlutoUI
	using Printf
end

# ╔═╡ ad1f0212-c412-11ee-220f-391425de7e32
md"# Predictive modelling of anticancer drug sensitivity in the Cancer Cell Line Encyclopedia

## Setup the environment

Activate the BINF301 environment:
"

# ╔═╡ 5ea50038-7c1d-407d-bb16-6a66490513b6
md"Load packages:"

# ╔═╡ 162e1bd3-1344-4f09-8a46-9d87a7b56d19
md"
## Load the data

Make sure that you have downloaded the data before running this notebook by executing the script `download_processed_data.jl` in the `scripts` folder. Information to load the data is stored in the `Data.toml` file, which needs to be loaded first:
"

# ╔═╡ 31678ef7-2d89-41c8-b2ad-7c46f5d3d9d1
DataSets.load_project!(projectdir("Data.toml"))

# ╔═╡ 0f233233-9760-471b-8655-3af9505e7500
tree = DataSets.open(dataset("CCLE"));

# ╔═╡ 84f0d28e-b7bd-439c-a0cf-f24d84ffa88d
df_sens = open(Vector{UInt8}, tree["CCLE-ActArea.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 2da47d13-1046-4df0-9c22-1db5445a4dfd
df_expr = open(Vector{UInt8}, tree["CCLE-expr.csv"]) do buf
           CSV.read(buf, DataFrame)
       end;

# ╔═╡ f8fa0306-e9b1-469d-89ba-ee114d8c4d72
md"""
## Predictive modelling using MLJ

We use the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) package and framework to train predictive models of drug sensitivity.

Read the [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/) for a quick overview of the package. If you're new(ish) to machine learning start with the [Learning MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_mlj/) page.
"""

# ╔═╡ 3279787c-4e25-4817-bae3-a6b065a8b07b
md"
We will use the elastic net regressin implementation of the [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) package. To read more about this regressor, remove the semi-colon at the end of the following command:
"

# ╔═╡ 2bef1d79-014e-42ec-b1b5-fe9ecf344a95
doc("ElasticNetRegressor",pkg="MLJLinearModels");

# ╔═╡ 1665c3fa-1be6-4359-ad1b-ec419f708d27
md"
Instantiate a model with default parameters:
"

# ╔═╡ fb53a425-5b1a-4a3c-bcb6-390757ef8f83
md"We will need the expression data in Matrix format for training models:"

# ╔═╡ 828f070c-c703-41a5-9fe4-3842e4627f5f
X = Matrix(df_expr);

# ╔═╡ 35fa5314-e7e3-4caf-b0d6-5bd888c578d6
histogram(mean(X,dims=1))

# ╔═╡ f7799446-46f7-40b1-92f1-914818691bcc
md"
### Modelling the response to PD-0325901

PD-0325901 is an [inhibitor](https://doi.org/10.1016/j.bmcl.2008.10.054), of the [mitogen-activated extracellular signal-regulated kinase (MEK) pathway](https://pubmed.ncbi.nlm.nih.gov/24121058/). A predictive model for sensitivity to PD-0325901 in the CCLE was derived in [Figure 2 of the CCLE paper](https://www.nature.com/articles/nature11003/figures/2). Here we train and evaluate a predictive model for PD-0325901 sensitivity using gene expression predictors only.

Create a vector with response data:
"

# ╔═╡ d0ab14b7-335a-4e02-9a3f-838b6b244be0
y = df_sens.:"PD-0325901";

# ╔═╡ 10cd0ee7-27de-4ff6-a7ba-3cdd80749133
# ╠═╡ show_logs = false
ElNetReg = @load "ElasticNetRegressor" pkg="MLJLinearModels"

# ╔═╡ 963c2d86-33f9-4bba-a1c8-3fc438c1f287
elnet = ElNetReg()

# ╔═╡ 904a9853-c85f-40c7-b326-492d2ea92e44
# ╠═╡ show_logs = false
mach = machine(elnet, X, y)

# ╔═╡ Cell order:
# ╟─ad1f0212-c412-11ee-220f-391425de7e32
# ╠═ada02ef3-ca52-4b63-a138-fa084904fd39
# ╠═57fffbda-1158-4f8c-9e58-46f19ac1276d
# ╟─5ea50038-7c1d-407d-bb16-6a66490513b6
# ╠═0aae0966-c1e1-4e7c-bfd7-711778cfb127
# ╟─162e1bd3-1344-4f09-8a46-9d87a7b56d19
# ╠═31678ef7-2d89-41c8-b2ad-7c46f5d3d9d1
# ╠═0f233233-9760-471b-8655-3af9505e7500
# ╠═84f0d28e-b7bd-439c-a0cf-f24d84ffa88d
# ╠═2da47d13-1046-4df0-9c22-1db5445a4dfd
# ╟─f8fa0306-e9b1-469d-89ba-ee114d8c4d72
# ╟─3279787c-4e25-4817-bae3-a6b065a8b07b
# ╠═2bef1d79-014e-42ec-b1b5-fe9ecf344a95
# ╟─1665c3fa-1be6-4359-ad1b-ec419f708d27
# ╠═963c2d86-33f9-4bba-a1c8-3fc438c1f287
# ╟─fb53a425-5b1a-4a3c-bcb6-390757ef8f83
# ╠═828f070c-c703-41a5-9fe4-3842e4627f5f
# ╠═35fa5314-e7e3-4caf-b0d6-5bd888c578d6
# ╟─f7799446-46f7-40b1-92f1-914818691bcc
# ╠═d0ab14b7-335a-4e02-9a3f-838b6b244be0
# ╠═10cd0ee7-27de-4ff6-a7ba-3cdd80749133
# ╠═904a9853-c85f-40c7-b326-492d2ea92e44
