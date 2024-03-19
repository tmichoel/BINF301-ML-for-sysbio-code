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

If all you ever want to do is fit Lasso or Elastic Net linear regression models, using packages such as [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) or [Lasso.jl](https://github.com/JuliaStats/Lasso.jl) is probably easiest. Here instead we will use the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine learning framework to train predictive models of drug sensitivity. [MLJ](https://github.com/alan-turing-institute/MLJ.jl) comes with a steeper learning curve, but a big reward in the form of a long list of [supported models](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/) that can be used with hardly any changes to the code below. A comparable machine learning framework for [R](https://www.r-project.org/) is [MLR3](https://mlr3.mlr-org.com/), and for [Python](https://www.python.org/) [scikit-learn](https://scikit-learn.org/stable/).

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

In general though, prediction should be based on a model where these hyperparameters are tuned for a specific dataset. We can follow the example in [Tuning models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/) in the MLJ documentation and tune the hyperparameter `min_purity_increase`, which determines how much a new decision tree split must improve the total score to be accepted.

First we define a range of values for the hyperparameter to be considered for tuning:
"

# ╔═╡ 22ffcf4f-20f2-4011-ab51-e151d37aa7f1
r = range(rf_model, :min_purity_increase, lower=0.001, upper=1.0, scale=:log);

# ╔═╡ 4e2f3a4a-e1ac-4f8d-bb01-ed2242ecaf0e
md"Then we have to define a `TunedModel` that will be used for the tuning. We need to specify the base model (Random Forest), how we will do the tuning (cross validation), how many values to test from our range object, and which measure we want to optimize:"

# ╔═╡ b339e2e3-a2f4-40a4-b264-750dd6ad18b8
self_tuning_rf_model = TunedModel(
    model=rf_model,
    resampling=CV(nfolds=3),
    tuning=Grid(resolution=10),
    range=r,
    measure=rms
);

# ╔═╡ e33459ea-1a81-4ca1-bbc1-a957e213b9ab
mach_tune_rf = machine(self_tuning_rf_model, X, y);

# ╔═╡ e34bf000-4f73-4508-b69a-318b88b66808
# ╠═╡ disabled = true
#=╠═╡
fit!(mach_tune_rf, rows=train)
  ╠═╡ =#

# ╔═╡ 9a6d7294-3e5a-451e-8464-3292c237a207
# ╠═╡ disabled = true
#=╠═╡
fitted_params(mach_tune_rf).best_model
  ╠═╡ =#

# ╔═╡ 8cc071d9-82fd-48f1-b84e-c1feceec607d
# ╠═╡ disabled = true
#=╠═╡
plot(mach_tune_rf)
  ╠═╡ =#

# ╔═╡ 8fb5a5a6-02c0-4c91-a959-c21c1da7d69b
# ╠═╡ disabled = true
#=╠═╡
yhat_tune_rf = MLJBase.predict(mach_tune_rf, rows=test);
  ╠═╡ =#

# ╔═╡ 7f833a57-7375-4373-abca-920de45ba85c
# ╠═╡ disabled = true
#=╠═╡
begin
	scatter(y[test],yhat_tune_rf, label="", xlabel="Real $(yname) values", ylabel="Predicted $(yname) values")
	title!(@sprintf("RMSE = %1.2f", rms(y[test],yhat_tune_rf)))
end
  ╠═╡ =#

# ╔═╡ 64caaf6d-92be-4b1f-938f-7d63d67acf33
md"
Since hyperparameter tuning is computationally expensive, we will not do it using the full set of $(size(X,2)) features (genes), but start with a feature selection step. In the [CCLE paper](https://doi.org/10.1038/nature11003), a prescreening was applied where only features were included that were correlated with the response vector with ``R > 0.1`` based on Pearson correlation. It is not clear from their text whether the prescreening was done on the whole data or not, but clearly, to prevent data leakage it must be done on the training data only. Move the slider to select a threshold on the absolute correlation between a gene and the drug sensitivity (keep the default of 0.25 if you want computations to be fast):
"

# ╔═╡ 159c60a3-7651-43a7-8e5f-6385752d809c
md"""
!!! note \"Note\"
	In the sections above, I emphasized the importance of separating models from data in MLJ. Shouldn't this prescreening step be part of the model instead of doing this manually and having to remember to select features using training data only? And even better, shouldn't the correlation threshold be a tunable/learnable parameter of the model itself?

	It turns out to be non-trivial to include a prescreening step that depends on both the features ``X`` and the target variable ``y``, requiring the use of so-called [Learning Networks](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_networks/) instead of the simple [Linear Pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/) we used before. If you're interested (and you should be!), see the **Advanced topic** section below.
"""

# ╔═╡ 21e3547e-2a77-4534-92df-f25f199d1f19


# ╔═╡ 3279787c-4e25-4817-bae3-a6b065a8b07b
md"
We will first use the elastic net regression implementation of the [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) package. To read more about this regressor, remove the semi-colon at the end of the following `doc` command. The most important point is the definition of the loss function. The `ElasticNetRegressor` (with default option `scale_penalty_with_samples=true`) finds the coefficients $\beta_j$ which minimize

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

# ╔═╡ ead508fc-3543-436c-becf-d6ebce50e37c
md"
Let's take the values for ``\alpha`` and ``\lambda'`` that we found in the notebook `Regularized_regression_Glmnet.jl` and compute the corresponding ``\lambda`` and ``\gamma``:
"

# ╔═╡ d67fb589-89e1-462f-841b-d2e94b72995a
α = 0.5

# ╔═╡ 301443c9-1465-4ad9-8b2d-17f5dcb0ebc9
λ_glmnet = 0.06

# ╔═╡ 477c4efc-bcfc-4a54-b143-71f8c17c637b
λ = λ_glmnet * (1-α)

# ╔═╡ 88514a70-c9c5-41f3-b474-9f41b92e9bc3
γ = λ_glmnet * α

# ╔═╡ e54510f4-0377-40b5-a30b-1893d98cd3e7
# ╠═╡ show_logs = false
ElasticNetRegressor = @load ElasticNetRegressor pkg="MLJLinearModels";

# ╔═╡ 2bef1d79-014e-42ec-b1b5-fe9ecf344a95
doc("ElasticNetRegressor",pkg="MLJLinearModels");

# ╔═╡ 707e0034-f707-4169-ac71-79489683153c
elnet_model = ElasticNetRegressor(lambda=λ, gamma=γ)

# ╔═╡ d0c78460-9a6d-493c-abee-08f5e249f2eb
md"
For elastic net regression, it is generally recommended to standardize features, but one must be careful: if the entire dataset is standardized before splitting into training and testing samples (a very common mistake!), information will have leaked from the train to the test data (because the mean and standard deviation are computed on the combined train and test data). In other words, **when evaluating a model fitted on standardized training data, the test data must be transformed using the means and standard deviations of the features in the training data!** When using a package like [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl), you as a user have to remember to do all this correctly yourself. The difficulty with this approach lies in the mixing between the *data* and how it is split for training and testing, and the *model*, which in this case is something like \"first standardize, then do elastic net regression\".

In [MLJ](https://github.com/alan-turing-institute/MLJ.jl), this is exactly how things work: we can define a model combining standardizing and regression by [composing models](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/):
"

# ╔═╡ fd58d84e-e22d-4c06-8d5e-63bd7aeb188d
std_elnet_model = Standardizer() |> elnet_model

# ╔═╡ 95379955-2fb1-452f-8ae1-37618c48397b
md"
### Fitting a model

To fit a model, we first have to couple it to data. In [MLJ](https://github.com/alan-turing-institute/MLJ.jl), this is done by defining a *machine*:
"

# ╔═╡ ad423168-c8dd-4e8b-b38d-835e6b2f6411
@bind threshold PlutoUI.Slider(0.1:0.05:1, default=0.25)

# ╔═╡ 122b36be-3759-4980-818e-b87fd05d7e95
selected_features = findall(map(x -> abs(cor(x,y[train])) > threshold, eachcol(X[train,:])));

# ╔═╡ ad902283-2f1c-4b03-9ee9-a46bfebe78a2
md"
A threshold of $(threshold) results in $(length(selected_features)) selected features. Define a new DataFrame to hold the selected features:
"

# ╔═╡ 553be392-0ea3-4b50-8647-da1251e1c04f
X_sub = X[:, selected_features];

# ╔═╡ e7c32cb9-e75a-4cc3-9c40-0496e88a65d5
mach = machine(std_elnet_model, X[:,selected_features], y);

# ╔═╡ 134a0be4-7cc6-4086-a0f9-8d7cd8a6be7e
md"
We can now fit our model to the data by calling:
"

# ╔═╡ 2ad90c8d-a123-4fd9-bd83-755667750635
# ╠═╡ show_logs = false
MLJBase.fit!(mach, rows=train);

# ╔═╡ b22b72e0-71f9-4568-8d0d-a05af800a786
md"
Finally we predict drug sensitivity values on the test samples and compare to their true values:
"

# ╔═╡ c6b77ed6-9f3a-4263-a8c7-c5170d0af702
# ╠═╡ show_logs = false
#=╠═╡
yhat = MLJ.predict(mach,rows=test);
  ╠═╡ =#

# ╔═╡ fac008e6-8d28-404f-ab74-6bfdca8e458c
#=╠═╡
begin
	scatter(y[test],yhat, label="", xlabel="Real $(yname) values", ylabel="Predicted $(yname) values")
	title!(@sprintf("RMSE = %1.2f", rms(y[test],yhat)))
end
  ╠═╡ =#

# ╔═╡ d75fa1b7-e5b2-4e24-a9d3-fd476457c570
md"
Let's start by only tuning the hyperparameter ``\gamma`` (``L_1`` penalty strength) of our `std_elnet_model` (standardization followed by Elastic Net) regressor, keeping ``\lambda`` at its default value. In the command below, note how the parameter being tuned is specified and how this differs from the [example in the MLJ docs](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-a-single-hyperparameter-using-a-grid-search-(regression-example)). Our model is a pipeline model, and hence we have to specify both the name of the hyperparameter being tuned and the component of the pipeline it belongs to. Also note that regularization strengths in Elastic Net regression are usually varied over a log-linear scale.
"

# ╔═╡ 23589a74-1049-482a-92a1-5dcdab01f700
# ╠═╡ disabled = true
#=╠═╡
gamma_range = range(std_elnet_model, :(elastic_net_regressor.gamma), lower=0.001, upper=2.0, scale=:log)
  ╠═╡ =#

# ╔═╡ 8236c3c0-decf-4d77-bbae-e2820c487965
md"
We can now define a tuning model to optimize the selected hyperparameter following the [example in the MLJ docs](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-a-single-hyperparameter-using-a-grid-search-(regression-example)). Using the `CV` option, the training samples are split in a number of ``k`` equal-sized \"folds\", and each of them is used in turn as a validation set, using the remaining ``k-1`` folds for training; the final measure for a particular ``\gamma``-value is the average root mean square error (`rms`) for the ``k`` predictions (here ``k`` is set to 5).
"

# ╔═╡ 8d907be8-e2fc-4463-b641-33c8302e9d2b
# ╠═╡ disabled = true
#=╠═╡
self_tuning_std_elnet_model = TunedModel(
    model=std_elnet_model,
    resampling=CV(nfolds=5),
    tuning=Grid(resolution=50, shuffle=false),
    range=gamma_range,
	acceleration=CPUThreads(),
    measure=rms
)
  ╠═╡ =#

# ╔═╡ d9e6c5c9-f91e-47fd-8f2b-e622214fd0a5
md"
Now tuning proceeds in the same way as before: we couple the tuning model to our data in a machine and fit the machine on the training samples:
"

# ╔═╡ 3a9f7d20-2d29-4bed-bbd1-ba51972c1649
# ╠═╡ disabled = true
#=╠═╡
mach_tuning = machine(self_tuning_std_elnet_model, X_sub, y);
  ╠═╡ =#

# ╔═╡ 9bc4ae61-0caf-4ca4-8302-64ef96ade4c0
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
fit!(mach_tuning, rows=train);
  ╠═╡ =#

# ╔═╡ e9572f8d-79be-4e5a-b24f-9749ab569fb4
md"
Here's another cool feature of MLJ, for a diagnostic plot of the `rms` values at each of the tested ``\gamma``-values, simply `plot` the tuning machine:
"

# ╔═╡ 59865b1d-ef8f-4a07-a011-3662f2fd2ea8
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
plot(mach_tuning)
  ╠═╡ =#

# ╔═╡ 6b9d71a4-0faa-4ed2-9adc-06f00bdf934d
# ╠═╡ disabled = true
#=╠═╡
gamma_best = report(mach_tuning).best_history_entry.model.elastic_net_regressor.gamma
  ╠═╡ =#

# ╔═╡ f19aab7c-79f8-46e4-a65c-74d625cfbace
md"
To predict drug sensitivities on the test samples using the best model found during hyperparameter tuning, call `predict` as before: 
"

# ╔═╡ 635cb960-6031-41df-8b10-e7e6429abd50
# ╠═╡ disabled = true
#=╠═╡
yhat_best = MLJBase.predict(mach_tuning, rows=test);
  ╠═╡ =#

# ╔═╡ 5d24acff-7011-4ffb-ab78-94be5bda7e67
# ╠═╡ disabled = true
#=╠═╡
begin
	scatter(y[test],yhat_best, label="", xlabel="Real $(yname) values", ylabel="Predicted $(yname) values")
	annotate!(6.,1.5,@sprintf("RMSE = %1.2f", rms(y[test],yhat_best)))
end
  ╠═╡ =#

# ╔═╡ 3b018ad8-a169-4b33-a853-f6dd65463032
md"""
Note our Elastic Net model has only two hyperparameters, such that exhaustive tuning over all hyperparameter value combinations in a pairwise grid is feasible. When tuning a large number of hyperparameters, more sophisticated sampling strategies have to be used, see for instance tuning using  [random search](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-using-a-random-search) or [latin hypercube sampling](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-using-Latin-hypercube-sampling).
"""

# ╔═╡ 1967638e-e1bf-4c6a-940d-01addd07309d
md"""
### Identifying important features

To identify important features in an elastic net model, we can inspect the regression coefficients. We can of course use the fitted machine from before, but if we are only interested in the important feature and not in prediction performance, we can also fit a model with the optimal ``\gamma`` parameter on the full data:
"""

# ╔═╡ 13151984-7281-4693-8ff8-cf5710ba9422
# ╠═╡ disabled = true
#=╠═╡
elnet_model_opt = ElasticNetRegressor(gamma=gamma_best)
  ╠═╡ =#

# ╔═╡ 37939bdf-0dfc-45e3-9132-0c422c42d78d
# ╠═╡ disabled = true
#=╠═╡
std_elnet_model_opt = Standardizer() |> elnet_model_opt;
  ╠═╡ =#

# ╔═╡ 45f57224-1b20-4bba-aef2-3e0b54223379
md"
Fit the model to the full data:
"

# ╔═╡ 3158d185-36c9-4373-957b-8ef15eedcc94
# ╠═╡ disabled = true
#=╠═╡
mach_opt = machine(std_elnet_model_opt, X, y);
  ╠═╡ =#

# ╔═╡ 37c17aaf-3742-4dae-91b3-25af8fcee927
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
fit!(mach_opt)
  ╠═╡ =#

# ╔═╡ 68683a4e-3b20-4cff-bfed-73650855b877
md"The coefficients are stored in the machine as a list of pairs:"

# ╔═╡ 7d702290-0c97-4495-980b-d7bb5368123c
# ╠═╡ disabled = true
#=╠═╡
fitted_params(mach_opt)
  ╠═╡ =#

# ╔═╡ 340a55a6-3cab-4d34-8cef-1cd7c4617b78
# ╠═╡ disabled = true
#=╠═╡
coefs = fitted_params(mach_opt).elastic_net_regressor.coefs
  ╠═╡ =#

# ╔═╡ 46e83648-1d8c-4e70-899f-82f6506bdddd
md"Let's put this in a DataFrame instead, also computing the absolute value as a measure of feature importance:"

# ╔═╡ 03a5c59d-130a-403b-95bd-24d68c7e629b
# ╠═╡ disabled = true
#=╠═╡
df_coefs = stack(DataFrame(coefs));
  ╠═╡ =#

# ╔═╡ d7387160-3694-44dc-a405-4a996d9b6610
#=╠═╡
DataFrames.select!(df_coefs, :variable, :value, :value => (x -> abs.(x)) => :abs_value);
  ╠═╡ =#

# ╔═╡ cfbfb033-cd61-498a-a5ed-7110aa04bd5c
md"Sort the coefficients by decreasing absolute value:"

# ╔═╡ 367bb570-5a5c-4029-bfa0-e3b96f92a13b
#=╠═╡
sort!(df_coefs, :abs_value, rev=true)
  ╠═╡ =#

# ╔═╡ 28d37c86-3c5d-4d8b-a5ae-c446ee136871
md"
In the CCLE paper, bootstrapping is performed using the optimal regularization strength to determine the robustness of selected features. The procedure samples training samples with replacement, relearns the model, and computes how frequently a feature (gene) is chosen as predictor in the bootstrapped models. Let's implement this procedure:
"

# ╔═╡ daa47734-80b5-416a-aa83-d9cfb35bb315
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
begin
	B = 200
	ns = nrow(X_sub);
	betas_bootstrap = DataFrames.select(df_coefs, :variable);# zeros(ncol(X_sub),B);
	for b = 1:B
		# Sample samples with replacement
	  	bootstrap_samples = sample(1:ns,ns,replace=true);
		# Fit the model to bootstrapped data
		Xb = X_sub[bootstrap_samples,:];
		yb = y[bootstrap_samples];
		machb = machine(std_elnet_model_opt, Xb, yb);
		fit!(machb)
		dfb = stack(DataFrame(fitted_params(machb).elastic_net_regressor.coefs))
		betas_bootstrap = innerjoin(betas_bootstrap, dfb, on=:variable, makeunique=true);
	end
end
  ╠═╡ =#

# ╔═╡ 7931d32b-a9cc-4f0e-bcfd-69238ad3b357
md"Compute the frequency with which each variable is selected to be part of a bootstrapped model:"

# ╔═╡ fd0ced0f-0ce1-44de-8978-dd49f54c7798
#=╠═╡
betas_nzfreq = DataFrame(:variable => betas_bootstrap.variable, :nonzero_freq => vec(mean(Matrix(betas_bootstrap[:,2:end]) .!= 0,dims=2)));
  ╠═╡ =#

# ╔═╡ 69f1e9c3-7a2a-4a32-b7dd-094dc926e250
md"Sort the result by non-zero frequency. Compare with the ranking based on regression coefficients for the optimal model on the full data:."

# ╔═╡ 282ae46f-2cb9-4031-8e3b-d26f3f9fc4a9
#=╠═╡
sort!(betas_nzfreq, :nonzero_freq, rev=true)
  ╠═╡ =#

# ╔═╡ 673b55ff-9c76-4dcd-86f8-f9f0fa1eeb71
#=╠═╡
md"
### Reproduce CCLE Figure 2

We keep all the genes with non-zero coefficients. You may have to set a higher threshold if you kept more features initially (at the start of the model tuning section). With the default settings of the notebook, we have only $(ncol(X_sub_filt)) features in the optimal model:
"
  ╠═╡ =#

# ╔═╡ fd88452d-321e-4678-bf65-f64bd8d5d06b
# ╠═╡ disabled = true
#=╠═╡
nms_filt = df_coefs.variable[df_coefs.abs_value .> 0.0]
  ╠═╡ =#

# ╔═╡ c8cf85e9-3a78-4bc8-acde-655453bce53e
#=╠═╡
X_sub_filt = DataFrames.select(X_sub, nms_filt)
  ╠═╡ =#

# ╔═╡ 72443bb9-fd89-4623-bb1b-6ccac35a3c1f
md"Standardize the filtered data:"

# ╔═╡ f960957c-759c-402a-858a-8e648e43b6d2
#=╠═╡
mach_std = machine(Standardizer(),X_sub_filt);
  ╠═╡ =#

# ╔═╡ ca58a5c2-1963-4960-a8f2-feffbb8cc118
#=╠═╡
fit!(mach_std);
  ╠═╡ =#

# ╔═╡ 7e9477c0-f203-4108-ab7b-4e48d90b2918
#=╠═╡
X_sub_filt_std = MLJ.transform(mach_std,X_sub_filt)
  ╠═╡ =#

# ╔═╡ ed27db95-3140-4efe-9b1e-78b2c81fceb7
md"Sort the samples by increasing levels of $(yname):"

# ╔═╡ f759997d-eb08-4555-a0cc-816e130c78bd
spy = sortperm(y);

# ╔═╡ 1287a9d6-cde2-4822-bf4d-caaa5871c234
md"For the heatmap, get the data of the selected features as a matrix with rows as features and columns as samples. Truncate values to make the range symmetric around zero."

# ╔═╡ 041a4dff-0bb0-4eeb-85dd-7b218f1cb85b
#=╠═╡
data_hm=Matrix(X_sub_filt_std[spy,:])';
  ╠═╡ =#

# ╔═╡ bfd70301-d292-4445-a778-1f0587000352
#=╠═╡
data_hm_mx = min(maximum(data_hm), abs(minimum(data_hm)))
  ╠═╡ =#

# ╔═╡ 34b3261c-3b5e-4688-8ff0-5fd227ea95f0
#=╠═╡
data_hm[data_hm .> data_hm_mx] .= data_hm_mx;
  ╠═╡ =#

# ╔═╡ db8365a3-db95-4e0e-bfdf-6e3eddd521ef
#=╠═╡
data_hm[data_hm .< -data_hm_mx] .= -data_hm_mx;
  ╠═╡ =#

# ╔═╡ a2af3fea-de83-49f8-b32a-6e40d7a41963
#=╠═╡
glab = names(X_sub_filt_std)
  ╠═╡ =#

# ╔═╡ f5ac398c-91de-40b7-a2d4-e7409707c43d
#=╠═╡
betas = sort(df_coefs.value[df_coefs.abs_value .> 0.0])
  ╠═╡ =#

# ╔═╡ b5df8a6a-c638-46b7-95ab-21dccf27091b
#=╠═╡
begin
	phm = heatmap(1:ns,glab,data_hm, c=:balance, xticks=[])
	
end
  ╠═╡ =#

# ╔═╡ f600462f-b68f-48f3-8f4e-b7b253db38b3
plot(y[spy], label="")

# ╔═╡ 70ef3d79-a202-4615-9849-9aa8a98a406d
#=╠═╡
pb = bar(0:1:length(betas)-1, -betas, label="", orientation = :h, xlims=[-maximum(betas),0], yticks=(0:1:length(betas)-1, glab))
  ╠═╡ =#

# ╔═╡ 62f15b33-78b9-4997-9f57-15859470f24d
#=╠═╡
begin
	l = @layout [a{0.3w} b{0.7w}]
	plot(pb, phm, layout=l)
end
  ╠═╡ =#

# ╔═╡ 87dfa45a-ae47-4a1e-a454-818073a25ad9
#=╠═╡
md"""
We can also see why Random Forest is popular. Out-of-the box it achieves comparable RMS error than the tuned Elastic Net model:  $(rms(y[test],yhat_rf)) vs. $( rms(y[test],yhat_best)).
"""
  ╠═╡ =#

# ╔═╡ c03647eb-d2e9-40f6-b627-a690284421e3
md"
## Extra topic: tuning more than one hyperparameter
To tune more than one hyperparameter, in the case of Elastic Net regression, tuning both regularization strengths ``\lambda`` and ``\gamma``, we need to define ranges for both hyperparameters. A range for ``\gamma`` was defined before, so we only need to define a range for ``\lambda`` here:
"

# ╔═╡ d0959d43-bfcd-44a4-9c12-eccd6f3f351c
# ╠═╡ disabled = true
#=╠═╡
lambda_range = range(std_elnet_model, :(elastic_net_regressor.lambda), lower=0.001, upper=2.0, scale=:log);
  ╠═╡ =#

# ╔═╡ bd356a7e-1c98-4a39-b7d3-22f3f49ef8d9
md"""
We now define another tuning model that performs a Grid search over the combined hyperparameter ranges:
"""

# ╔═╡ 75906b07-22b4-445f-9c91-ac41146baae8
# ╠═╡ disabled = true
#=╠═╡
self_tuning_std_elnet_model_2 = TunedModel(
    model=std_elnet_model,
    resampling=CV(nfolds=5),
    tuning=Grid(resolution=20, shuffle=false),
    range=[lambda_range, gamma_range],
	acceleration=CPUThreads(),
    measure=rms
)
  ╠═╡ =#

# ╔═╡ 05d34211-6a55-453d-b401-e2eb00bae662
md"A machine is created as before:"

# ╔═╡ f24aadbd-f33b-4d52-a86c-381e50b2045f
# ╠═╡ disabled = true
#=╠═╡
mach_tuning_2 = machine(self_tuning_std_elnet_model_2, X_sub, y);
  ╠═╡ =#

# ╔═╡ a0e72ad6-06dc-43df-84f2-43f9bd40be82
md"Fitting this machine will take more time (**Why?**). You can try it by enabling the following cell:"

# ╔═╡ aab73e6d-6ea5-429f-8248-09d51d69ba2a
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
fit!(mach_tuning_2, rows=train);
  ╠═╡ =#

# ╔═╡ 02aebe3c-8af9-482c-ba3e-33703b518139
md"""
The tuning results can again be visualized using the `plot` command. If you have fitted the machine in the previous step, you can enable the following cells to make the diagnostic plot and extract the best ``\lambda`` and ``\gamma`` values. What do the color and size of the markers in the bottom left panel of the plot mean?
"""

# ╔═╡ 476dde61-56cb-48a2-a80f-b12d50a30f7b
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
plot(mach_tuning_2)
  ╠═╡ =#

# ╔═╡ e2234adf-2532-4ebe-af62-e82cde4d134e
# ╠═╡ disabled = true
#=╠═╡
lambda_best = report(mach_tuning_2).best_history_entry.model.elastic_net_regressor.lambda
  ╠═╡ =#

# ╔═╡ 6df55699-2008-424a-9d92-8aa4f25d3a2c
md"
## Advanced topic: Learning Networks

See the [Learning networks (1)](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks/) and [Learning networks (2)](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/learning-networks-2/) tutorials
"

# ╔═╡ fc42330a-be6f-42d2-b1c5-a263c104d266
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 6b8673a3-d293-42c1-a9b3-84207eb5ba7e
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 68b42a23-eeaa-4fa3-83c8-122bf0b62793
# ╠═╡ disabled = true
#=╠═╡
fs = FeatureSelectorCor()
  ╠═╡ =#

# ╔═╡ c9f0cbc5-3723-499e-bcb6-5fd8f0db3eeb
# ╠═╡ disabled = true
#=╠═╡
ms = machine(fs,X,y);
  ╠═╡ =#

# ╔═╡ ba3e5eb9-3137-4a0d-9f90-017068aab34b
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
fit!(ms)
  ╠═╡ =#

# ╔═╡ f931bee8-e370-4cb4-ad0f-6a82b8c5c7eb
# ╠═╡ disabled = true
#=╠═╡
size(MLJ.transform(ms,X))
  ╠═╡ =#

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
m2 = machine(std_elnet_model,x,ys);
  ╠═╡ =#

# ╔═╡ d448c154-d54a-48dd-acc7-da33994f29f7
# ╠═╡ disabled = true
#=╠═╡
yhat_ln = MLJBase.predict(m2,x);
  ╠═╡ =#

# ╔═╡ 5885192b-7922-4785-bb84-e1220574c988
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
fit!(yhat_ln,rows=train);
  ╠═╡ =#

# ╔═╡ e479f2b2-4e5d-41c3-90d0-518d0bf783b2
# ╠═╡ disabled = true
#=╠═╡
yhat_ln(rows=train)
  ╠═╡ =#

# ╔═╡ 945b6a9d-03b0-4328-85c9-1b331c9f67fc
# ╠═╡ disabled = true
#=╠═╡
begin
	scatter(y[test],yhat_ln(rows=test), label="", xlabel="Real $(yname) values", ylabel="Predicted $(yname) values")
	annotate!(4.5,0,@sprintf("RMSE = %1.2f", rms(y[test],yhat_ln(rows=test))))
end
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
# ╠═f8fa0306-e9b1-469d-89ba-ee114d8c4d72
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
# ╠═22ffcf4f-20f2-4011-ab51-e151d37aa7f1
# ╟─4e2f3a4a-e1ac-4f8d-bb01-ed2242ecaf0e
# ╠═b339e2e3-a2f4-40a4-b264-750dd6ad18b8
# ╠═e33459ea-1a81-4ca1-bbc1-a957e213b9ab
# ╠═e34bf000-4f73-4508-b69a-318b88b66808
# ╠═9a6d7294-3e5a-451e-8464-3292c237a207
# ╠═8cc071d9-82fd-48f1-b84e-c1feceec607d
# ╠═8fb5a5a6-02c0-4c91-a959-c21c1da7d69b
# ╠═7f833a57-7375-4373-abca-920de45ba85c
# ╠═64caaf6d-92be-4b1f-938f-7d63d67acf33
# ╠═ad902283-2f1c-4b03-9ee9-a46bfebe78a2
# ╠═553be392-0ea3-4b50-8647-da1251e1c04f
# ╟─159c60a3-7651-43a7-8e5f-6385752d809c
# ╠═21e3547e-2a77-4534-92df-f25f199d1f19
# ╟─3279787c-4e25-4817-bae3-a6b065a8b07b
# ╠═ead508fc-3543-436c-becf-d6ebce50e37c
# ╠═d67fb589-89e1-462f-841b-d2e94b72995a
# ╠═301443c9-1465-4ad9-8b2d-17f5dcb0ebc9
# ╠═477c4efc-bcfc-4a54-b143-71f8c17c637b
# ╠═88514a70-c9c5-41f3-b474-9f41b92e9bc3
# ╠═e54510f4-0377-40b5-a30b-1893d98cd3e7
# ╠═2bef1d79-014e-42ec-b1b5-fe9ecf344a95
# ╠═707e0034-f707-4169-ac71-79489683153c
# ╠═d0c78460-9a6d-493c-abee-08f5e249f2eb
# ╠═fd58d84e-e22d-4c06-8d5e-63bd7aeb188d
# ╟─95379955-2fb1-452f-8ae1-37618c48397b
# ╠═ad423168-c8dd-4e8b-b38d-835e6b2f6411
# ╠═122b36be-3759-4980-818e-b87fd05d7e95
# ╠═e7c32cb9-e75a-4cc3-9c40-0496e88a65d5
# ╟─134a0be4-7cc6-4086-a0f9-8d7cd8a6be7e
# ╠═2ad90c8d-a123-4fd9-bd83-755667750635
# ╟─b22b72e0-71f9-4568-8d0d-a05af800a786
# ╠═c6b77ed6-9f3a-4263-a8c7-c5170d0af702
# ╠═fac008e6-8d28-404f-ab74-6bfdca8e458c
# ╟─d75fa1b7-e5b2-4e24-a9d3-fd476457c570
# ╠═23589a74-1049-482a-92a1-5dcdab01f700
# ╟─8236c3c0-decf-4d77-bbae-e2820c487965
# ╠═8d907be8-e2fc-4463-b641-33c8302e9d2b
# ╟─d9e6c5c9-f91e-47fd-8f2b-e622214fd0a5
# ╠═3a9f7d20-2d29-4bed-bbd1-ba51972c1649
# ╠═9bc4ae61-0caf-4ca4-8302-64ef96ade4c0
# ╟─e9572f8d-79be-4e5a-b24f-9749ab569fb4
# ╠═59865b1d-ef8f-4a07-a011-3662f2fd2ea8
# ╠═6b9d71a4-0faa-4ed2-9adc-06f00bdf934d
# ╟─f19aab7c-79f8-46e4-a65c-74d625cfbace
# ╠═635cb960-6031-41df-8b10-e7e6429abd50
# ╠═5d24acff-7011-4ffb-ab78-94be5bda7e67
# ╟─3b018ad8-a169-4b33-a853-f6dd65463032
# ╟─1967638e-e1bf-4c6a-940d-01addd07309d
# ╠═13151984-7281-4693-8ff8-cf5710ba9422
# ╠═37939bdf-0dfc-45e3-9132-0c422c42d78d
# ╟─45f57224-1b20-4bba-aef2-3e0b54223379
# ╠═3158d185-36c9-4373-957b-8ef15eedcc94
# ╠═37c17aaf-3742-4dae-91b3-25af8fcee927
# ╟─68683a4e-3b20-4cff-bfed-73650855b877
# ╠═7d702290-0c97-4495-980b-d7bb5368123c
# ╠═340a55a6-3cab-4d34-8cef-1cd7c4617b78
# ╟─46e83648-1d8c-4e70-899f-82f6506bdddd
# ╠═03a5c59d-130a-403b-95bd-24d68c7e629b
# ╠═d7387160-3694-44dc-a405-4a996d9b6610
# ╟─cfbfb033-cd61-498a-a5ed-7110aa04bd5c
# ╠═367bb570-5a5c-4029-bfa0-e3b96f92a13b
# ╟─28d37c86-3c5d-4d8b-a5ae-c446ee136871
# ╠═daa47734-80b5-416a-aa83-d9cfb35bb315
# ╟─7931d32b-a9cc-4f0e-bcfd-69238ad3b357
# ╠═fd0ced0f-0ce1-44de-8978-dd49f54c7798
# ╟─69f1e9c3-7a2a-4a32-b7dd-094dc926e250
# ╠═282ae46f-2cb9-4031-8e3b-d26f3f9fc4a9
# ╟─673b55ff-9c76-4dcd-86f8-f9f0fa1eeb71
# ╠═fd88452d-321e-4678-bf65-f64bd8d5d06b
# ╠═c8cf85e9-3a78-4bc8-acde-655453bce53e
# ╟─72443bb9-fd89-4623-bb1b-6ccac35a3c1f
# ╠═f960957c-759c-402a-858a-8e648e43b6d2
# ╠═ca58a5c2-1963-4960-a8f2-feffbb8cc118
# ╠═7e9477c0-f203-4108-ab7b-4e48d90b2918
# ╟─ed27db95-3140-4efe-9b1e-78b2c81fceb7
# ╠═f759997d-eb08-4555-a0cc-816e130c78bd
# ╟─1287a9d6-cde2-4822-bf4d-caaa5871c234
# ╠═041a4dff-0bb0-4eeb-85dd-7b218f1cb85b
# ╠═bfd70301-d292-4445-a778-1f0587000352
# ╠═34b3261c-3b5e-4688-8ff0-5fd227ea95f0
# ╠═db8365a3-db95-4e0e-bfdf-6e3eddd521ef
# ╠═a2af3fea-de83-49f8-b32a-6e40d7a41963
# ╠═f5ac398c-91de-40b7-a2d4-e7409707c43d
# ╠═b5df8a6a-c638-46b7-95ab-21dccf27091b
# ╠═f600462f-b68f-48f3-8f4e-b7b253db38b3
# ╠═70ef3d79-a202-4615-9849-9aa8a98a406d
# ╠═62f15b33-78b9-4997-9f57-15859470f24d
# ╟─87dfa45a-ae47-4a1e-a454-818073a25ad9
# ╟─c03647eb-d2e9-40f6-b627-a690284421e3
# ╠═d0959d43-bfcd-44a4-9c12-eccd6f3f351c
# ╟─bd356a7e-1c98-4a39-b7d3-22f3f49ef8d9
# ╠═75906b07-22b4-445f-9c91-ac41146baae8
# ╟─05d34211-6a55-453d-b401-e2eb00bae662
# ╠═f24aadbd-f33b-4d52-a86c-381e50b2045f
# ╟─a0e72ad6-06dc-43df-84f2-43f9bd40be82
# ╠═aab73e6d-6ea5-429f-8248-09d51d69ba2a
# ╟─02aebe3c-8af9-482c-ba3e-33703b518139
# ╠═476dde61-56cb-48a2-a80f-b12d50a30f7b
# ╠═e2234adf-2532-4ebe-af62-e82cde4d134e
# ╟─6df55699-2008-424a-9d92-8aa4f25d3a2c
# ╠═fc42330a-be6f-42d2-b1c5-a263c104d266
# ╠═6b8673a3-d293-42c1-a9b3-84207eb5ba7e
# ╠═68b42a23-eeaa-4fa3-83c8-122bf0b62793
# ╠═c9f0cbc5-3723-499e-bcb6-5fd8f0db3eeb
# ╠═ba3e5eb9-3137-4a0d-9f90-017068aab34b
# ╠═f931bee8-e370-4cb4-ad0f-6a82b8c5c7eb
# ╠═11acff61-f6fb-4f75-a85a-149aff1f7f8e
# ╠═28cb414b-da63-4c0a-998f-f98623b2e9c9
# ╠═cef7b439-0307-4286-a3d5-b406f53fc1a5
# ╠═fddd2e38-7b68-4ba7-930e-95faaf8ac51f
# ╠═8d4cec94-6a7f-4fad-a6f2-bb63e68e5dbd
# ╠═d448c154-d54a-48dd-acc7-da33994f29f7
# ╠═5885192b-7922-4785-bb84-e1220574c988
# ╠═e479f2b2-4e5d-41c3-90d0-518d0bf783b2
# ╠═945b6a9d-03b0-4328-85c9-1b331c9f67fc
# ╟─459c65de-a914-4d9c-992e-386afa5e5d2b
