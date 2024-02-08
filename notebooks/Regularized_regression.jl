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
	using MLJModels
	using MLJBase
	using BetaML
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
X = open(Vector{UInt8}, tree["CCLE-expr.csv"]) do buf
           CSV.read(buf, DataFrame)
       end;

# ╔═╡ f7799446-46f7-40b1-92f1-914818691bcc
md"
### Modelling the response to PD-0325901

PD-0325901 is an [inhibitor](https://doi.org/10.1016/j.bmcl.2008.10.054), of the [mitogen-activated extracellular signal-regulated kinase (MEK) pathway](https://pubmed.ncbi.nlm.nih.gov/24121058/). A predictive model for sensitivity to PD-0325901 in the CCLE was derived in [Figure 2 of the CCLE paper](https://www.nature.com/articles/nature11003/figures/2). Here we train and evaluate predictive models for PD-0325901 sensitivity using gene expression predictors only.

Create a vector with response data:
"

# ╔═╡ d0ab14b7-335a-4e02-9a3f-838b6b244be0
y = Array(df_sens.:"PD-0325901");

# ╔═╡ f8fa0306-e9b1-469d-89ba-ee114d8c4d72
md"""
## Predictive modelling using MLJ

If all you ever want to do is fit Lasso or Elastic Net linear regression models, using packages such as [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) or [Lasso.jl](https://github.com/JuliaStats/Lasso.jl) is probably easiest. [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) in particular wraps around the same Fortran code underlying GLMNet packages in other languages (such as the [glmnet](https://www.jstatsoft.org/article/view/v033i01) package for R) and has the same user interface.

Here instead we will use the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine learning framework to train predictive models of drug sensitivity. [MLJ](https://github.com/alan-turing-institute/MLJ.jl) comes with a steeper learning curve, but a big reward in the form of a long list of [supported models](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/) that can be used instead of Elastic Net with hardly any changes to the code below.

Read the [Getting Started](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/) for a quick overview of [MLJ](https://github.com/alan-turing-institute/MLJ.jl). If you're new(ish) to machine learning start with the [Learning MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_mlj/) page.
"""

# ╔═╡ 3279787c-4e25-4817-bae3-a6b065a8b07b
md"
We will use the elastic net regression implementation of the [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) package. To read more about this regressor, remove the semi-colon at the end of the following `doc` command. The most important point is the definition of the loss function. The `ElasticNetRegressor` (with default option `scale_penalty_with_samples=true`) finds the coefficients $\beta_j$ which minimize

$\frac12\sum_{i=1}^n \Bigl(y_i - \beta_0 - \sum_{j=1}^p X_{ij}\beta_j\Bigr)^2 + \frac{n\lambda}2 \|\beta\|_2^2 + n \gamma \|\beta\|_1$

where $\|\beta\|_2^2 = \sum_{j=1}^p \beta_j^2$ and $\|\beta\|_1 = \sum_{j=1}^p |\beta_j|$.

By contrast, in the [CCLE paper](https://doi.org/10.1038/nature11003) (and GLMNet package),  the loss function is parameterized as:

$\frac1{2n}\sum_{i=1}^n \Bigl(y_i - \beta_0 - \sum_{j=1}^p X_{ij}\beta_j\Bigr)^2 + \lambda' \Bigl(\frac{(1-\alpha)}2 \|\beta\|_2^2 + \alpha \|\beta\|_1\Bigr)$

Hence to compare hyperparameter values, we have to remember

```math
\begin{aligned}
\lambda &= \lambda'(1-\alpha)\\
\gamma &= \lambda'\alpha
\end{aligned}
```

or the other way around:

```math
\begin{aligned}
\lambda' &= \lambda + \gamma\\
\alpha &= \frac\gamma{\lambda+\gamma}
\end{aligned}
```

Also note the `ElasticNetRegressor` default values of $\lambda=1.0$ and $\gamma=0.0$, i.e. by default the `ElasticNetRegressor` loss function penalty does not include the $L_1$ term.
"

# ╔═╡ 2bef1d79-014e-42ec-b1b5-fe9ecf344a95
doc("ElasticNetRegressor",pkg="MLJLinearModels");

# ╔═╡ bb437a19-be1d-4145-8dc1-be53584b69fb
md"
### Setting up the model

A key aspect of [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is the separation between *model* and *data*.

It is generally recommended to standardize features for elastic net regression, but one must be careful: if the entire dataset is standardized before splitting into training and testing samples, information will have leaked from the train to the test data (because the mean and standard deviation are computed on the combined train and test data). In other words, when evaluating a model fitted on standardized training data, the test data must be transformed using the means and standard deviations of the features in the training data! In MLJ, we can accomplish this quite neatly by [composing models](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/).

When an elastic net regression model is fitted to centred data (features having mean zero) and the intercept term is not penalized, the intercept estimate will be the mean (over the training samples) of the target variable. Hence there is not really a need to standardize the target variable, but to illustrate how this is done in MLJ (using a [target transformation](https://alan-turing-institute.github.io/MLJ.jl/dev/target_transformations/)), we do it anyway.


"

# ╔═╡ 064c11f6-97f0-4d84-87da-a8bea80ac8ee
# ╠═╡ disabled = true
#=╠═╡
md"
Finally, in the [CCLE paper](https://doi.org/10.1038/nature11003), a prescreening was applied where only features were included that were correlated with the response vector with $R > 0.1$ based on Pearson correlation. It is not clear from their text whether the prescreening was done on the whole data or not, but clearly, to prevent data leakage it must be done on the training data only. Hence we need a step in our pipeline that performs this prescreening. The solution is to create a custom [static transformer](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Static-transformers):
";
  ╠═╡ =#

# ╔═╡ 75b939b3-de26-4b37-9143-65d629c2b794
# ╠═╡ disabled = true
#=╠═╡
begin
	mutable struct Prescreen <: Static
	    threshold::Float64
	end
	
	function MLJ.transform(transformer::Prescreen, _, X, y)
	    selected_features = [feature for feature in 1:size(X,2) if abs(cor(X[!, feature], y)) >= transformer.threshold]
	    return X[:,selected_features]
	end
	
end
  ╠═╡ =#

# ╔═╡ fca1439a-ec0b-4afc-8b00-7dfcb1cfefb6
md"
We can now create a [linear pipeline](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/) consisting of a prescreening step, followed by a standardizer, and followed by an elastic net regressor, wrapped in a [TransformedTargetModel](https://alan-turing-institute.github.io/MLJ.jl/dev/target_transformations/#MLJBase.TransformedTargetModel)
"

# ╔═╡ 7ac06cb5-c604-41a4-bc10-65587414c07c
elnet_std = Standardizer() |> ElasticNetRegressor()

# ╔═╡ 0e426484-b4a2-4488-8838-9991d75c0512
elnet_std2 = TransformedTargetModel(elnet_std , transformer=Standardizer())

# ╔═╡ 5406c51f-488d-4db6-97c4-191f31a31173
md"
We will train a model on 80% of the samples and test it on the remaining 20%:
"

# ╔═╡ b76c66be-75b6-402b-9b3b-eca64b585cf1
train, test = partition(shuffle(eachindex(y)), 0.8);

# ╔═╡ 52b4aed2-6a96-436f-a4ed-c8ba8f0a00a5
selected_features = [feature for feature in 1:size(X,2) if abs(cor(X[train, feature], y[train])) >= 0.1]

# ╔═╡ 0fe32c35-ca36-4596-b6f8-29a02219f6b2
length(selected_features)

# ╔═╡ eb00ec56-2a2f-4e34-8a1f-64e2e27412dc
md"
We can now bind our model pipeline to the data:
"

# ╔═╡ d4c3dcb4-c9a3-44da-be0f-3fb0401e5475
mach2 = machine( RidgeRegressor(), Z[:,selected_features], y);

# ╔═╡ 459c65de-a914-4d9c-992e-386afa5e5d2b
# ╠═╡ disabled = true
#=╠═╡
begin
	mutable struct CorrelationFilter <: MLJBase.Deterministic
	    threshold::Float64
	end
	
	CorrelationFilter(;threshold=0.5) = CorrelationFilter(threshold)
	
	function MLJBase.fit(model::CorrelationFilter, verbosity, X, y)
		Xm = MLJBase.matrix(X)
	    correlations = abs.(cor(Xm, y))
	    valid_features = findall(correlations .> model.threshold)
		cache = nothing
		report = nothing
	    return valid_features, cache, report
	end
	
	function MLJBase.transform(model::CorrelationFilter, fitresult, Xnew)
	    return Xnew[:, fitresult.valid_features]
	end
end
  ╠═╡ =#

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
# ╠═f7799446-46f7-40b1-92f1-914818691bcc
# ╠═d0ab14b7-335a-4e02-9a3f-838b6b244be0
# ╟─f8fa0306-e9b1-469d-89ba-ee114d8c4d72
# ╟─3279787c-4e25-4817-bae3-a6b065a8b07b
# ╠═2bef1d79-014e-42ec-b1b5-fe9ecf344a95
# ╠═bb437a19-be1d-4145-8dc1-be53584b69fb
# ╠═064c11f6-97f0-4d84-87da-a8bea80ac8ee
# ╠═75b939b3-de26-4b37-9143-65d629c2b794
# ╟─fca1439a-ec0b-4afc-8b00-7dfcb1cfefb6
# ╠═7ac06cb5-c604-41a4-bc10-65587414c07c
# ╠═0e426484-b4a2-4488-8838-9991d75c0512
# ╟─5406c51f-488d-4db6-97c4-191f31a31173
# ╠═b76c66be-75b6-402b-9b3b-eca64b585cf1
# ╠═52b4aed2-6a96-436f-a4ed-c8ba8f0a00a5
# ╠═0fe32c35-ca36-4596-b6f8-29a02219f6b2
# ╟─eb00ec56-2a2f-4e34-8a1f-64e2e27412dc
# ╠═d4c3dcb4-c9a3-44da-be0f-3fb0401e5475
# ╟─459c65de-a914-4d9c-992e-386afa5e5d2b
