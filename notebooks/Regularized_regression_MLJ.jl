### A Pluto.jl notebook ###
# v0.19.41

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
	using MLJBase
	using MLJModelInterface
	using Statistics
	using StatsBase
	using StatsPlots, LaTeXStrings
	using PlutoUI
	using Printf
end

# ╔═╡ ad1f0212-c412-11ee-220f-391425de7e32
md"# Predictive modelling of anticancer drug sensitivity in the Cancer Cell Line Encyclopedia using MLJ

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
# ╠═╡ show_logs = false
X = open(Vector{UInt8}, tree["CCLE-expr.csv"]) do buf
           CSV.read(buf, DataFrame)
       end;

# ╔═╡ f7799446-46f7-40b1-92f1-914818691bcc
md"
### Modelling the response to PD-0325901

PD-0325901 is an [inhibitor](https://doi.org/10.1016/j.bmcl.2008.10.054), of the [mitogen-activated extracellular signal-regulated kinase (MEK) pathway](https://pubmed.ncbi.nlm.nih.gov/24121058/). A predictive model for sensitivity to PD-0325901 in the CCLE was derived in [Figure 2 of the CCLE paper](https://www.nature.com/articles/nature11003/figures/2). Here we train and evaluate predictive models for PD-0325901 sensitivity using gene expression predictors only.

Create a vector with response data:
"

# ╔═╡ f2dc3074-1d80-4ace-84ad-8b9199c2c769
yname = "PD-0325901";

# ╔═╡ 123b5c83-b43f-4e78-8583-0976056d2afa
y = Array(df_sens.:"PD-0325901");

# ╔═╡ f8fa0306-e9b1-469d-89ba-ee114d8c4d72
md"""
## Predictive modelling using MLJ

If all you ever want to do is fit Lasso or Elastic Net linear regression models, using packages such as [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) or [Lasso.jl](https://github.com/JuliaStats/Lasso.jl) is easiest. Here instead we will use the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine learning framework to train non-linear predictive models of drug sensitivity. [MLJ](https://github.com/alan-turing-institute/MLJ.jl) comes with a steeper learning curve, but a big reward in the form of a long list of [supported models](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/) that can be used with hardly any changes to the code below. A comparable machine learning framework for [R](https://www.r-project.org/) is [MLR3](https://mlr3.mlr-org.com/), and for [Python](https://www.python.org/) [scikit-learn](https://scikit-learn.org/stable/).

Read the [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/) for a quick overview of [MLJ](https://github.com/alan-turing-institute/MLJ.jl). If you're new(ish) to machine learning start with the [Learning MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_mlj/) page.

[Random forest regression](https://en.wikipedia.org/wiki/Random_forest) is a popular non-linear model, so let's see how it does on our data. To read more, remove the semi-colon at the end of the `doc` statement.
"""

# ╔═╡ 20d69d51-4182-470c-9b88-e0da43d1cc16
# ╠═╡ show_logs = false
RandomForestRegressor = @load RandomForestRegressor pkg="DecisionTree";

# ╔═╡ 55253db7-4410-4fab-adb5-f82aa705d3d9
doc("RandomForestRegressor",pkg="DecisionTree");

# ╔═╡ bb437a19-be1d-4145-8dc1-be53584b69fb
md"
### Setting up the model

A key aspect of [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is the separation between *model* and *data*. For regression problems $y=f(x_1,\dots,x_p)$, the *model* describes the family of functions $f$ being considered. Defining a model in [MLJ](https://github.com/alan-turing-institute/MLJ.jl) can be as simple as:
"

# ╔═╡ 387cd0e4-6887-4ebf-ab8f-59607e01d15f
rf_model = RandomForestRegressor();

# ╔═╡ 0f0dae8f-150d-4382-8b73-34cb42b5e705
md"
### Fitting the model

We will fit a random forest with default hyperparameters. Because a random forest is a set of [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning), and the nodes in a decision tree test whether a certain feature is above or below a learned threshold value, standardization of the features will make no difference and is not needed in this case.

In [MLJ](https://github.com/alan-turing-institute/MLJ.jl), a model is coupled to the data in a *machine*:
"

# ╔═╡ 26192120-5742-4416-b568-7b6bbd37c0c9
mach_rf = machine(rf_model, X, y);

# ╔═╡ 4db0ffa4-1ffd-4c01-ae4b-b2617d2bccb3
md"
We typically fit a model to a subset of \"training\" samples and test its performance on the remaining \"test\" samples. Randomly partitioning the samples into 80% training and 20% test samples:
"

# ╔═╡ b76c66be-75b6-402b-9b3b-eca64b585cf1
train, test = MLJBase.partition(eachindex(y), 0.8; shuffle=true, rng=123);

# ╔═╡ 6964c19a-e101-4920-b630-a5009058e88e
fit!(mach_rf, rows=train);

# ╔═╡ f60d6ee9-b4ee-4b92-b5bc-a2958972e860
yhat_rf = MLJBase.predict(mach_rf, rows=test);

# ╔═╡ 7e55d699-8cbe-4bd4-8bb0-eb45c1e21e70
md"Compare predicted and true values:"

# ╔═╡ 182890e8-1fd1-4f29-92cb-c1f1d2b9480b
begin
	scatter(y[test],yhat_rf, label="", xlabel="Real $(yname) values", ylabel="Predicted $(yname) values")
	title!(@sprintf("RMSE = %1.2f", rms(y[test],yhat_rf)))
end

# ╔═╡ e66465f8-857b-4942-a28c-f152a82d7a31
md"
We can check the feature importances of the fitted model:
"

# ╔═╡ bd2df233-3789-4f3d-afba-87a2da63403c
df_fimp = stack(DataFrame(feature_importances(mach_rf)))

# ╔═╡ 7d6c3437-361a-4d3e-87ef-b04029417950
md"
### Tuning a model

So far, we have fitted a Random Forest regression model using default hyperparameter values. Random Forests are popular because these defaults in general give good prediction performance - compare the RMSE in the figure above to that of the tuned Elastic Net model in the `Regularized_regression_Glmnet.jl` notebook.

In general though, prediction should be based on a model where these hyperparameters are tuned for a specific dataset. See the [Tuning models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/) page in the MLJ documentation for details.
"

# ╔═╡ 5b3a085e-28eb-4894-be46-cf8e286e8571
md"
### Exercise

In the model above we used *all* features (genes) as predictors in the model. Can you fit a Random Forest regressor that only uses features with absolute correlation greater than 0.1, similar to the `Regularized_regression_Glmnet.jl` notebook? Remember that feature selection must be done on the training samples only to prevent data leakage!
"

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
# ╟─f7799446-46f7-40b1-92f1-914818691bcc
# ╠═f2dc3074-1d80-4ace-84ad-8b9199c2c769
# ╠═123b5c83-b43f-4e78-8583-0976056d2afa
# ╟─f8fa0306-e9b1-469d-89ba-ee114d8c4d72
# ╠═20d69d51-4182-470c-9b88-e0da43d1cc16
# ╠═55253db7-4410-4fab-adb5-f82aa705d3d9
# ╟─bb437a19-be1d-4145-8dc1-be53584b69fb
# ╠═387cd0e4-6887-4ebf-ab8f-59607e01d15f
# ╟─0f0dae8f-150d-4382-8b73-34cb42b5e705
# ╠═26192120-5742-4416-b568-7b6bbd37c0c9
# ╟─4db0ffa4-1ffd-4c01-ae4b-b2617d2bccb3
# ╠═b76c66be-75b6-402b-9b3b-eca64b585cf1
# ╠═6964c19a-e101-4920-b630-a5009058e88e
# ╠═f60d6ee9-b4ee-4b92-b5bc-a2958972e860
# ╟─7e55d699-8cbe-4bd4-8bb0-eb45c1e21e70
# ╠═182890e8-1fd1-4f29-92cb-c1f1d2b9480b
# ╟─e66465f8-857b-4942-a28c-f152a82d7a31
# ╠═bd2df233-3789-4f3d-afba-87a2da63403c
# ╟─7d6c3437-361a-4d3e-87ef-b04029417950
# ╟─5b3a085e-28eb-4894-be46-cf8e286e8571
