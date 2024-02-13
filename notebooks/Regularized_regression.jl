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

# ╔═╡ e54510f4-0377-40b5-a30b-1893d98cd3e7
# ╠═╡ show_logs = false
ElasticNetRegressor = @load ElasticNetRegressor pkg="MLJLinearModels";

# ╔═╡ 2bef1d79-014e-42ec-b1b5-fe9ecf344a95
doc("ElasticNetRegressor",pkg="MLJLinearModels");

# ╔═╡ bb437a19-be1d-4145-8dc1-be53584b69fb
md"
### Setting up the model

A key aspect of [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is the separation between *model* and *data*. For regression problems $y=f(x_1,\dots,x_p)$, the *model* describes the family of functions $f$ being considered. Defining a model in [MLJ](https://github.com/alan-turing-institute/MLJ.jl) can be as simple as:
"

# ╔═╡ 707e0034-f707-4169-ac71-79489683153c
elnet_model = ElasticNetRegressor(lambda=0.2, gamma=0.2)

# ╔═╡ d0c78460-9a6d-493c-abee-08f5e249f2eb
md"
For elastic net regression, it is generally recommended to standardize features, but one must be careful: if the entire dataset is standardized before splitting into training and testing samples, information will have leaked from the train to the test data (because the mean and standard deviation are computed on the combined train and test data). In other words, when evaluating a model fitted on standardized training data, the test data must be transformed using the means and standard deviations of the features in the training data! When using a package like [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl), you as a user have to remember to do all this correctly yourself. The difficulty with this approach lies in the mixing between the *data* and how it is split for training and testing, and the *model*, which in this case is something like \"first standardize, then do elastic net regression\".

In [MLJ](https://github.com/alan-turing-institute/MLJ.jl), this is exactly how things work: we can define a model combining standardizing and regression by [composing models](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/):
"

# ╔═╡ fd58d84e-e22d-4c06-8d5e-63bd7aeb188d
std_elnet_model = Standardizer() |> elnet_model

# ╔═╡ 95379955-2fb1-452f-8ae1-37618c48397b
md"
### Fitting a model

To fit a model, we first have to couple it to data. In [MLJ](https://github.com/alan-turing-institute/MLJ.jl), this is done by defining a *machine*:
"

# ╔═╡ e7c32cb9-e75a-4cc3-9c40-0496e88a65d5
mach = machine(std_elnet_model, X, y);

# ╔═╡ 4db0ffa4-1ffd-4c01-ae4b-b2617d2bccb3
md"
We typically fit a model to a subset of \"training\" samples and test its performance on the remaining \"test\" samples. Randomly partitioning the samples into 80% training and 20% test samples:
"

# ╔═╡ b76c66be-75b6-402b-9b3b-eca64b585cf1
train, test = MLJBase.partition(eachindex(y), 0.8; shuffle=true);

# ╔═╡ 134a0be4-7cc6-4086-a0f9-8d7cd8a6be7e
md"
We can now fit our model to the data by calling:
"

# ╔═╡ 2ad90c8d-a123-4fd9-bd83-755667750635
# ╠═╡ show_logs = false
MLJBase.fit!(mach, rows=train, force=true);

# ╔═╡ b22b72e0-71f9-4568-8d0d-a05af800a786
md"
Finally we predict drug sensitivity values on the test samples and compare to their true values:
"

# ╔═╡ 7d6c3437-361a-4d3e-87ef-b04029417950
md"
### Tuning a model

So far, we have fitted an Elastic Net regression model using arbitrary hyperparameter values. Of course, prediction should be based on a model where these hyperparameters are tuned for a specific dataset. See [Tuning models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/) in the MLJ documentation.
"

# ╔═╡ 64caaf6d-92be-4b1f-938f-7d63d67acf33
md"
Since hyperparameter tuning is computationally expensive, we will not do it using the full set of $(size(X,2)) features (genes), but start with a feature selection step. In the [CCLE paper](https://doi.org/10.1038/nature11003), a prescreening was applied where only features were included that were correlated with the response vector with ``R > 0.1`` based on Pearson correlation. It is not clear from their text whether the prescreening was done on the whole data or not, but clearly, to prevent data leakage it must be done on the training data only. Move the slider to select a threshold on the absolute correlation between a gene and the drug sensitivity:
"

# ╔═╡ ad423168-c8dd-4e8b-b38d-835e6b2f6411
@bind threshold Slider(0:0.05:1, default=0.25)

# ╔═╡ 122b36be-3759-4980-818e-b87fd05d7e95
selected_features = findall(map(x -> abs(cor(x,y[train])) > threshold, eachcol(X[train,:])));

# ╔═╡ ad902283-2f1c-4b03-9ee9-a46bfebe78a2
md"
A threshold of $(threshold) results in $(length(selected_features)) selected features. Define a new DataFrame to hold the selected features:
"

# ╔═╡ 553be392-0ea3-4b50-8647-da1251e1c04f
X_sub = X[:, selected_features];

# ╔═╡ 159c60a3-7651-43a7-8e5f-6385752d809c
md"""
!!! note \"Note\"
	In the sections above, I emphasized the importance of separating models from data in MLJ. Shouldn't this prescreening step be part of the model instead of doing this manually and having to remember to select features using training data only? And even better, shouldn't the correlation threshold be a tunable/learnable parameter of the model itself?

	It turns out to be non-trivial to include a prescreening step that depends on both the features ``X`` and the target variable ``y``, requiring the use of so-called [Learning Networks](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_networks/) instead of the simple [Linear Pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/) we used before. If you're interested (and you should be!), see the **Advanced topic** section below.
"""

# ╔═╡ d75fa1b7-e5b2-4e24-a9d3-fd476457c570
md"
Let's start by only tuning the hyperparameter ``\gamma`` (``L_1`` penalty strength) of our `std_elnet_model` (standardization followed by Elastic Net) regressor. In the command below, note how the parameter being tuned is specified and how this differs from the [example in the MLJ docs](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-a-single-hyperparameter-using-a-grid-search-(regression-example)). Our model is a pipeline model, and hence we have to specify both the name of the hyperparameter being tuned and the component of the pipeline it belongs to. Also note that regularization strengths in Elastic Net regression are usually varied over a log-linear scale.
"

# ╔═╡ 23589a74-1049-482a-92a1-5dcdab01f700
gamma_range = range(std_elnet_model, :(elastic_net_regressor.gamma), lower=0.001, upper=1.0, scale=:log)

# ╔═╡ 8236c3c0-decf-4d77-bbae-e2820c487965
md"
We can now define a tuning model to optimize the selected hyperparameter (keeping the ``L_2`` regularization parameter ``\lambda`` fixed) following the [example in the MLJ docs](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-a-single-hyperparameter-using-a-grid-search-(regression-example)). To save some computation time we set the `resampling` parameter to `Holdout` instead of `CV` (cross-validation). Using the `Holdout` option, for each ``\gamma``-value being evaluated, 70% of the training samples are randomly selected for fitting the model and the remaining 30% are used fro validating it, using root mean square error (`rms`) as a measure. Using the `CV` option, the training samples are split in a number of ``k`` equal-sized \"folds\", and each of them is used in turn as a validation set, using the remaining ``k-1`` folds for training; the final measure for a particular ``\gamma``-value is the average `rms` for the ``k`` predictions.
"

# ╔═╡ 8d907be8-e2fc-4463-b641-33c8302e9d2b
self_tuning_std_elnet_model = TunedModel(
    model=std_elnet_model,
    resampling=Holdout(shuffle=true),
    tuning=Grid(resolution=50, shuffle=false),
    range=gamma_range,
	acceleration=CPUThreads(),
    measure=rms
)

# ╔═╡ d9e6c5c9-f91e-47fd-8f2b-e622214fd0a5
md"
Now tuning proceeds in the same way as before: we couple the tuning model to our data in a machine and fit the machine on the training samples:
"

# ╔═╡ 3a9f7d20-2d29-4bed-bbd1-ba51972c1649
mach_tuning = machine(self_tuning_std_elnet_model, X_sub, y);

# ╔═╡ 9bc4ae61-0caf-4ca4-8302-64ef96ade4c0
# ╠═╡ show_logs = false
fit!(mach_tuning, rows=train);

# ╔═╡ e9572f8d-79be-4e5a-b24f-9749ab569fb4
md"
Here's another cool feature of MLJ, for a diagnostic plot of the `rms` values at each of the tested ``\gamma``-values, simply `plot` the tuning machine:
"

# ╔═╡ 59865b1d-ef8f-4a07-a011-3662f2fd2ea8
plot(mach_tuning)

# ╔═╡ f19aab7c-79f8-46e4-a65c-74d625cfbace
md"
To predict drug sensitivities on the test samples using the best model found during hyperparameter tuning, call `predict` as before: 
"

# ╔═╡ b62249e0-2b70-4721-96e9-fb66205f20da
md"
### Trying other models

As mentioned before, the power of using a framework like [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) is the availability of a large number of [models](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/) that can be fitted using a uniform syntax. [Random forest regression](https://en.wikipedia.org/wiki/Random_forest) is a popular non-linear model, so let's see how it does on our data. To read more, remove the semi-colon at the end of the `doc` statement.
"

# ╔═╡ 20d69d51-4182-470c-9b88-e0da43d1cc16
# ╠═╡ show_logs = false
RandomForestRegressor = @load RandomForestRegressor pkg="BetaML";

# ╔═╡ 55253db7-4410-4fab-adb5-f82aa705d3d9
doc("RandomForestRegressor",pkg="BetaML");

# ╔═╡ 0f0dae8f-150d-4382-8b73-34cb42b5e705
md"
We will fit a random forest with default hyperparameters. Because a random forest is a set of [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning), and the nodes in a decision tree test whether a certain feature is above or below a learned threshold value, standardization of the features will make no difference and is not needed in this case. 
"

# ╔═╡ 387cd0e4-6887-4ebf-ab8f-59607e01d15f
rf_model = RandomForestRegressor()

# ╔═╡ f0427f12-57fb-4632-96be-304b858545fd
md"Now couple the model to the data in a machine and fit the model on the training samples. To save computing time, let's use the smaller dataset with preselected features."

# ╔═╡ 86fc4031-b342-48da-9e92-263b69cb03b8
mach_rf = machine(rf_model, X_sub, y);

# ╔═╡ 8ec5eaee-5f39-4633-a6cf-777aba872b26
fit!(mach_rf, rows=train);

# ╔═╡ 7e55d699-8cbe-4bd4-8bb0-eb45c1e21e70
md"Compare predicted and true values:"

# ╔═╡ 6427f4e8-e232-43ae-84a2-6c789468da67
md"Compare values predicted by the best Elastic Net model and random forest model:"

# ╔═╡ 6df55699-2008-424a-9d92-8aa4f25d3a2c
md"
## Advanced topic: Learning Networks

See the [Learning networks (1)](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks/) and [Learning networks (2)](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/) tutorials
"

# ╔═╡ fc42330a-be6f-42d2-b1c5-a263c104d266
begin
	MLJModelInterface.@mlj_model mutable struct FeatureSelectorStd <: Unsupervised
	    threshold::Float64 = 1.0::(_ > 0)
	end
		
	function MLJ.fit(fs::FeatureSelectorStd, verbosity, X)
		# Find the standard deviation of each feature
	    stds = map(x -> std(x), eachcol(X))
		selected_features = findall(stds .> fs.threshold)
		cache = nothing
		report = nothing
		return selected_features, cache, nothing
	end
	
	function MLJ.transform(::FeatureSelectorStd, fitresult, X)
	    # Return the selected features
	    return selectcols(X,fitresult)
	end
end

# ╔═╡ 6b8673a3-d293-42c1-a9b3-84207eb5ba7e
begin
	MLJModelInterface.@mlj_model mutable struct FeatureSelectorCor <: Unsupervised
	    threshold::Float64 = 0.1::(_ > 0)
	end
	#FeatureSelectorCor(; threshold=0.1) = FeatureSelectorCor(threshold)
	
	function MLJ.fit(fs::FeatureSelectorCor, verbosity, X, y)
		# Find the correlation of each feature with y
	    cors = map(x -> abs(cor(x,y)), eachcol(X))
		selected_features = findall(cors .> fs.threshold)
		cache = nothing
		report = nothing
		return selected_features, cache, nothing
	end
	
	function MLJ.predict(::FeatureSelectorCor, fitresult, X, y)
	    # Return the selected features
	    return selectcols(X,fitresult)
	end

	function MLJ.transform(::FeatureSelectorCor, fitresult, X)
	    # Return the selected features
	    return selectcols(X,fitresult)
	end
end

# ╔═╡ c6b77ed6-9f3a-4263-a8c7-c5170d0af702
yhat = MLJ.predict(mach,rows=test);

# ╔═╡ fac008e6-8d28-404f-ab74-6bfdca8e458c
scatter(y[test],yhat)

# ╔═╡ 635cb960-6031-41df-8b10-e7e6429abd50
yhat_best = predict(mach_tuning, rows=test);

# ╔═╡ f6281ade-48c9-4d29-8fd1-217ac6081c1c
scatter(y[test],yhat_best)

# ╔═╡ 7e1f10bd-8b39-49e0-be75-e0108cc70c7a
yhat_rf = predict(mach_rf, rows=test);

# ╔═╡ 6ff71ca3-35bd-477c-9b1f-0e4343addf01
scatter(y[test],yhat_rf)

# ╔═╡ f75d7d99-8abd-4a64-9432-62540bedb9aa
scatter(yhat_best,yhat_rf)

# ╔═╡ 87dfa45a-ae47-4a1e-a454-818073a25ad9
md"""
We can also see why Random Forest is popular. Out-of-the box it achieves a lower RMS error than the tuned Elastic Net model:  $(rms(y[test],yhat_rf)) vs. $( rms(y[test],yhat_best)).
"""

# ╔═╡ 11acff61-f6fb-4f75-a85a-149aff1f7f8e
# ╠═╡ disabled = true
#=╠═╡
Xs = source(X);
  ╠═╡ =#

# ╔═╡ 28cb414b-da63-4c0a-998f-f98623b2e9c9
# ╠═╡ disabled = true
#=╠═╡
ys = source(y);
  ╠═╡ =#

# ╔═╡ cef7b439-0307-4286-a3d5-b406f53fc1a5
# ╠═╡ disabled = true
#=╠═╡
m1 = machine(FeatureSelectorCor(threshold=0.35),Xs, ys);
  ╠═╡ =#

# ╔═╡ fddd2e38-7b68-4ba7-930e-95faaf8ac51f
# ╠═╡ disabled = true
#=╠═╡
x = MLJ.transform(m1,Xs);
  ╠═╡ =#

# ╔═╡ 8d4cec94-6a7f-4fad-a6f2-bb63e68e5dbd
# ╠═╡ disabled = true
#=╠═╡
m2 = machine(ElasticNetRegressor(lambda=0.2,gamma=0.2),x,ys);
  ╠═╡ =#

# ╔═╡ d448c154-d54a-48dd-acc7-da33994f29f7
# ╠═╡ disabled = true
#=╠═╡
yhat = predict(m2,x);
  ╠═╡ =#

# ╔═╡ 5885192b-7922-4785-bb84-e1220574c988
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
fit!(yhat,rows=train);
  ╠═╡ =#

# ╔═╡ e479f2b2-4e5d-41c3-90d0-518d0bf783b2
# ╠═╡ disabled = true
#=╠═╡
size(yhat(rows=test));
  ╠═╡ =#

# ╔═╡ c1dbcb88-4a29-4c05-b9f7-0934b1db2066
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
mach = machine(mdl3, X, y);
  ╠═╡ =#

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
# ╠═ad1f0212-c412-11ee-220f-391425de7e32
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
# ╠═d0ab14b7-335a-4e02-9a3f-838b6b244be0
# ╟─f8fa0306-e9b1-469d-89ba-ee114d8c4d72
# ╟─3279787c-4e25-4817-bae3-a6b065a8b07b
# ╠═e54510f4-0377-40b5-a30b-1893d98cd3e7
# ╠═2bef1d79-014e-42ec-b1b5-fe9ecf344a95
# ╟─bb437a19-be1d-4145-8dc1-be53584b69fb
# ╠═707e0034-f707-4169-ac71-79489683153c
# ╟─d0c78460-9a6d-493c-abee-08f5e249f2eb
# ╠═fd58d84e-e22d-4c06-8d5e-63bd7aeb188d
# ╟─95379955-2fb1-452f-8ae1-37618c48397b
# ╠═e7c32cb9-e75a-4cc3-9c40-0496e88a65d5
# ╟─4db0ffa4-1ffd-4c01-ae4b-b2617d2bccb3
# ╠═b76c66be-75b6-402b-9b3b-eca64b585cf1
# ╟─134a0be4-7cc6-4086-a0f9-8d7cd8a6be7e
# ╠═2ad90c8d-a123-4fd9-bd83-755667750635
# ╟─b22b72e0-71f9-4568-8d0d-a05af800a786
# ╠═c6b77ed6-9f3a-4263-a8c7-c5170d0af702
# ╠═fac008e6-8d28-404f-ab74-6bfdca8e458c
# ╟─7d6c3437-361a-4d3e-87ef-b04029417950
# ╟─64caaf6d-92be-4b1f-938f-7d63d67acf33
# ╠═ad423168-c8dd-4e8b-b38d-835e6b2f6411
# ╠═122b36be-3759-4980-818e-b87fd05d7e95
# ╟─ad902283-2f1c-4b03-9ee9-a46bfebe78a2
# ╠═553be392-0ea3-4b50-8647-da1251e1c04f
# ╟─159c60a3-7651-43a7-8e5f-6385752d809c
# ╟─d75fa1b7-e5b2-4e24-a9d3-fd476457c570
# ╠═23589a74-1049-482a-92a1-5dcdab01f700
# ╟─8236c3c0-decf-4d77-bbae-e2820c487965
# ╠═8d907be8-e2fc-4463-b641-33c8302e9d2b
# ╟─d9e6c5c9-f91e-47fd-8f2b-e622214fd0a5
# ╠═3a9f7d20-2d29-4bed-bbd1-ba51972c1649
# ╠═9bc4ae61-0caf-4ca4-8302-64ef96ade4c0
# ╟─e9572f8d-79be-4e5a-b24f-9749ab569fb4
# ╠═59865b1d-ef8f-4a07-a011-3662f2fd2ea8
# ╟─f19aab7c-79f8-46e4-a65c-74d625cfbace
# ╠═635cb960-6031-41df-8b10-e7e6429abd50
# ╠═f6281ade-48c9-4d29-8fd1-217ac6081c1c
# ╟─b62249e0-2b70-4721-96e9-fb66205f20da
# ╠═20d69d51-4182-470c-9b88-e0da43d1cc16
# ╠═55253db7-4410-4fab-adb5-f82aa705d3d9
# ╟─0f0dae8f-150d-4382-8b73-34cb42b5e705
# ╠═387cd0e4-6887-4ebf-ab8f-59607e01d15f
# ╟─f0427f12-57fb-4632-96be-304b858545fd
# ╠═86fc4031-b342-48da-9e92-263b69cb03b8
# ╠═8ec5eaee-5f39-4633-a6cf-777aba872b26
# ╠═7e1f10bd-8b39-49e0-be75-e0108cc70c7a
# ╟─7e55d699-8cbe-4bd4-8bb0-eb45c1e21e70
# ╠═6ff71ca3-35bd-477c-9b1f-0e4343addf01
# ╟─6427f4e8-e232-43ae-84a2-6c789468da67
# ╠═f75d7d99-8abd-4a64-9432-62540bedb9aa
# ╟─87dfa45a-ae47-4a1e-a454-818073a25ad9
# ╟─6df55699-2008-424a-9d92-8aa4f25d3a2c
# ╠═fc42330a-be6f-42d2-b1c5-a263c104d266
# ╠═6b8673a3-d293-42c1-a9b3-84207eb5ba7e
# ╠═11acff61-f6fb-4f75-a85a-149aff1f7f8e
# ╠═28cb414b-da63-4c0a-998f-f98623b2e9c9
# ╠═cef7b439-0307-4286-a3d5-b406f53fc1a5
# ╠═fddd2e38-7b68-4ba7-930e-95faaf8ac51f
# ╠═8d4cec94-6a7f-4fad-a6f2-bb63e68e5dbd
# ╠═d448c154-d54a-48dd-acc7-da33994f29f7
# ╠═5885192b-7922-4785-bb84-e1220574c988
# ╠═e479f2b2-4e5d-41c3-90d0-518d0bf783b2
# ╠═c1dbcb88-4a29-4c05-b9f7-0934b1db2066
# ╟─459c65de-a914-4d9c-992e-386afa5e5d2b
