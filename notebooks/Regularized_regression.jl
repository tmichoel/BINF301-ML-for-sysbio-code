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
Y = open(Vector{UInt8}, tree["CCLE-ActArea.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ 2da47d13-1046-4df0-9c22-1db5445a4dfd
X = open(Vector{UInt8}, tree["CCLE-expr.csv"]) do buf
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

# ╔═╡ bb437a19-be1d-4145-8dc1-be53584b69fb
md"
It is generally recommended to standardize features for elastic net regression, but one must be careful: if the entire dataset is standardized before splitting into training and testing samples, information will have leaked from the train to the test data (because the mean and standard deviation are computed on the combined train and test data). In other words, when evaluating a model fitted on standardized training data, the test data must be transformed using the means and standard deviations of the features in the training data! In MLJ, we can accomplish this quite neatly by [composing models](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/).

When an elastic net regression model is fitted to centred data (features having mean zero) and the intercept term is not penalized, the intercept estimate will be the mean (over the training samples) of the target variable. Hence there is not really a need to standardize the target variable, but to illustrate how this is done in MLJ (using a [target transformation](https://alan-turing-institute.github.io/MLJ.jl/dev/target_transformations/), we do it anyway.

Hence we create a [linear pipeline](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/) consisting of a standardizer followed by an elastic net regressor, wrapped in a [TransformedTargetModel](https://alan-turing-institute.github.io/MLJ.jl/dev/target_transformations/#MLJBase.TransformedTargetModel):
"

# ╔═╡ 0e426484-b4a2-4488-8838-9991d75c0512
elnet_std = TransformedTargetModel( Pipeline(Standardizer(), ElasticNetRegressor()) , transformer=Standardizer())

# ╔═╡ f7799446-46f7-40b1-92f1-914818691bcc
md"
### Modelling the response to PD-0325901

PD-0325901 is an [inhibitor](https://doi.org/10.1016/j.bmcl.2008.10.054), of the [mitogen-activated extracellular signal-regulated kinase (MEK) pathway](https://pubmed.ncbi.nlm.nih.gov/24121058/). A predictive model for sensitivity to PD-0325901 in the CCLE was derived in [Figure 2 of the CCLE paper](https://www.nature.com/articles/nature11003/figures/2). Here we train and evaluate a predictive model for PD-0325901 sensitivity using gene expression predictors only.

Create a vector with response data:
"

# ╔═╡ d0ab14b7-335a-4e02-9a3f-838b6b244be0
y = Array(Y.:"PD-0325901");

# ╔═╡ eb00ec56-2a2f-4e34-8a1f-64e2e27412dc
md"
We can now bind our model pipeline to the data:
"

# ╔═╡ 904a9853-c85f-40c7-b326-492d2ea92e44
# ╠═╡ show_logs = false
mach = machine(elnet_std, X, y);

# ╔═╡ 5406c51f-488d-4db6-97c4-191f31a31173
md"
We will train a model on 80% of the samples and test it on the remaining 20%:
"

# ╔═╡ b76c66be-75b6-402b-9b3b-eca64b585cf1
train, test = partition(shuffle(eachindex(y)), 0.8);

# ╔═╡ 98d2c3c8-2b55-468b-8791-aed3cc718335
fit!(mach, rows=train);

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
# ╟─bb437a19-be1d-4145-8dc1-be53584b69fb
# ╠═0e426484-b4a2-4488-8838-9991d75c0512
# ╟─f7799446-46f7-40b1-92f1-914818691bcc
# ╠═d0ab14b7-335a-4e02-9a3f-838b6b244be0
# ╟─eb00ec56-2a2f-4e34-8a1f-64e2e27412dc
# ╠═904a9853-c85f-40c7-b326-492d2ea92e44
# ╟─5406c51f-488d-4db6-97c4-191f31a31173
# ╠═b76c66be-75b6-402b-9b3b-eca64b585cf1
# ╠═98d2c3c8-2b55-468b-8791-aed3cc718335
