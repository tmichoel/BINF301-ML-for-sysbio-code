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

# ╔═╡ 44b0fb80-c13b-490a-a28c-f45fe69ffe97
using DrWatson

# ╔═╡ 0bd0d1d8-0bdb-4755-b3db-b1885de15ae4
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ c6664290-bc06-48ae-b586-9ea2f8af5d36
# ╠═╡ show_logs = false
begin
	using DataFrames
	using CSV
	using DataSets
	using Statistics
	using StatsBase
	using StatsPlots, LaTeXStrings
	using GLMNet
	using MLJ
	using Printf
	using PlutoUI
end

# ╔═╡ 0b0c3970-de94-11ed-1198-03fb54d95a60
md"# Predictive modelling of anticancer drug sensitivity in the Cancer Cell Line Encyclopedia using GLMNet

## Setup the environment

Activate the BINF301 environment:
"

# ╔═╡ 96844648-bf72-4232-ab8e-3e552c476ab3
md"
## Load the data

Make sure that you have downloaded the data before running this notebook by executing the script `download_processed_data.jl` in the `scripts` folder. Information to load the data is stored in the `Data.toml` file, which needs to be loaded first:
"

# ╔═╡ 8e4a6f00-b274-48b4-870e-cf0cba0e20ff
DataSets.load_project!(projectdir("Data.toml"))

# ╔═╡ fc0d9eb1-ed41-46c6-b5a1-9ff35fa1789d
tree = DataSets.open(dataset("CCLE"));

# ╔═╡ 46b51265-37fa-47d1-b494-b4c8564380b8
df_sens = open(Vector{UInt8}, tree["CCLE-ActArea.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ e006f801-89e1-4b3a-8680-6fc458819877
X = open(Vector{UInt8}, tree["CCLE-expr.csv"]) do buf
           CSV.read(buf, DataFrame)
       end;

# ╔═╡ 71ba55fa-280b-4993-a70b-da0763d0a4c1
size(X)

# ╔═╡ a391c536-44a9-4a06-9f8f-79bf60275018
md"
### Modelling the response to PD-0325901

PD-0325901 is an [inhibitor](https://doi.org/10.1016/j.bmcl.2008.10.054), of the [mitogen-activated extracellular signal-regulated kinase (MEK) pathway](https://pubmed.ncbi.nlm.nih.gov/24121058/). A predictive model for sensitivity to PD-0325901 in the CCLE was derived in [Figure 2 of the CCLE paper](https://www.nature.com/articles/nature11003/figures/2). Here we train and evaluate predictive models for PD-0325901 sensitivity using gene expression predictors only.

Create a vector with response data:
"

# ╔═╡ 005ae139-18ac-47a6-a7bd-a84a4e8d72d1
yname = "PD-0325901";

# ╔═╡ fa8c02f1-70f5-42d4-acb7-791cc3cae92d
y = select(df_sens, yname)[:,1];

# ╔═╡ 613688c8-2113-43c7-83d9-25f137212ca9
md"""
## Predictive modelling using GLMNet

If all you ever want to do is fit Lasso or Elastic Net linear regression models, using packages such as [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) or [Lasso.jl](https://github.com/JuliaStats/Lasso.jl) is easiest. [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) in particular wraps around the same Fortran code underlying GLMNet packages in other languages (such as the [glmnet](https://www.jstatsoft.org/article/view/v033i01) package for R) and has the same user interface. If you want to fit more general, non-linear models, using a machine learning framework like [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is recommended. In this notebook we use [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl). For an example using [MLJ](https://github.com/alan-turing-institute/MLJ.jl), see the notebook `Regularized_regression_MLJ.jl`.
"""

# ╔═╡ 8a000026-152e-44f9-a2f9-f6d44cf3250e
md"First we split the data in training and test samples. Note that we set the random seed such that we can reproduce the exact same partitioning in the other notebook."

# ╔═╡ 92a37123-b262-4cd4-80b2-fa75a8d51b29
train, test = partition(eachindex(y), 0.8; shuffle=true, rng=123);

# ╔═╡ 88521c4f-cd8d-484f-a4b6-4e735702fb9d
md"
In the [CCLE paper](https://doi.org/10.1038/nature11003), a prescreening was applied where only features were included that were correlated with the response vector with ``R > 0.1`` based on Pearson correlation. It is not clear from their text whether the prescreening was done on the whole data or not, but clearly, to prevent data leakage it must be done on the training data only. Move the slider to select a threshold on the absolute correlation between a gene and the drug sensitivity (default value 0.1):"

# ╔═╡ 0c1bc2c8-cbc5-4af2-94ba-004dd7272e57
@bind cccut Slider(0:0.05:0.5, default=0.1)

# ╔═╡ 3ed5819f-b5f7-413e-a50d-71faa4a9bc0e
cc = map(x -> abs(cor(x,y[train])), eachcol(X[train,:]));

# ╔═╡ ae6af865-a992-4aee-a1d4-c4db8127378a
begin
	histogram(cc, xlabel="Correlation with drug sensitivity", label="")
	vline!([cccut],linewidth=2,label="")
end

# ╔═╡ f2b1a375-236e-429a-b3f9-64dea50f9a2f
md"Reduce predictor data and keep indices of selected genes"

# ╔═╡ 39e157a7-432e-485e-8f42-4152077c52a2
begin
	gene_select = cc.>cccut; #sd.>sdcut .&& cc.>cccut;
	Xtrain_select = Matrix(X[train,gene_select]); 
	Xtest_select = Matrix(X[test,gene_select]);
	names_select = names(X[train,gene_select]);
end;

# ╔═╡ 9d2246c6-8d68-449c-b258-58eb914e46f8
md"
For Elastic Net regression, it is generally recommended to standardize features, but one must be careful: if the entire dataset is standardized before splitting into training and testing samples (a very common mistake!), information will have leaked from the train to the test data (because the mean and standard deviation are computed on the combined train and test data). In other words, **when evaluating a model fitted on standardized training data, the test data must be transformed using the means and standard deviations of the features in the training data!**

Create a standardizer for the training data and apply it to both train and test data:"

# ╔═╡ 826b0d23-bcf7-433c-82f7-428a39a044e0
begin
	dt = StatsBase.fit(ZScoreTransform, Xtrain_select; dims=1)
	Xtrain_select_std = StatsBase.transform(dt,Xtrain_select)
	Xtest_select_std = StatsBase.transform(dt,Xtest_select)
end;

# ╔═╡ e071ecf4-2441-4bf5-8830-9546428e27fa
md"Now we define an Elastic Net cross-validation model. After a bit of playing around I didn't see much difference in performance when using multiple values for the L1 ratio parameter ($\alpha$), or using the 250 values for the regularization strength paramater ($\lambda$) as in the CCLE paper. Hence, for simplicity anmd reduced computation time, we use a fixed value for the L1 ratio (default 0.5), the default 100  automatically determined regularization strength paramater values, and the default 10-fold cross-validation. Move the slider to change the value of the L1 ratio from 0 (ridge regression) to 1 (lasso regression).
"

# ╔═╡ 03aff45f-206d-44ad-a587-1a8113056f85
@bind α Slider(0:0.1:1, default=0.5) 

# ╔═╡ 6519f119-9f91-4ca3-b9ec-d9116149cae3
cv = glmnetcv(Xtrain_select_std, y[train]; 
  alpha=α,
  standardize=false,
  )

# ╔═╡ caabc7ea-2f25-4a88-9621-133e384a08d0
md"Show the mean and standard deviation of the loss across the 10 folds, with the regularization strength on log-scale:"

# ╔═╡ 2fe8a865-00c4-431b-a070-916d0d31a070
begin
	plot(cv.lambda,cv.meanloss, 
	  xscale=:log10, 
	  legend=false, 
	  yerror=cv.stdloss,
	  xlabel=L"\lambda",
	  ylabel="loss")
	vline!([lambdamin(cv)])
end

# ╔═╡ a6a903a9-9b06-436b-a656-36c71a282117
md"Now train a new elastic net model on the training data with this regularization strength to find the final coefficient estimates:"

# ╔═╡ 35dbc788-430b-4491-a792-a369fb81b0a5
mdl = glmnet(Xtrain_select_std, y[train]; 
  alpha=α, 
  lambda=[lambdamin(cv)],
  standardize=false)

# ╔═╡ f7092f32-b332-4a61-80c4-46c92e33f7c7
md"Test the trained model on the test data."

# ╔═╡ 469000fe-d028-4e2f-93c2-fff727d373bd
ypred = GLMNet.predict(mdl, Xtest_select_std);

# ╔═╡ fff9698d-06a5-4e75-9ada-734806b66370
begin
	scatter(y[test],ypred,
	  label = "",
	  xlabel = "True drug sensitivities",
	  ylabel = "Predicted drug sensitivities"
	)
	title!(Printf.@sprintf("RMS = %1.2f", rms(y[test],vec(ypred))))
end

# ╔═╡ b284165c-e351-4a7a-980a-aa66b676de8b
md"## Bootstrapping
In the CCLE paper, bootstrapping is performed using the optimal regularization strength to determine the robustness of selected features. The procedure samples training samples with replacement, relearns the model, and computes how frequently a feature (gene) is chosen as predictor in the bootstrapped models."

# ╔═╡ 18ec73f8-9286-4f24-91d1-16596332ae9c
X_select = Matrix(X[:,gene_select]);

# ╔═╡ 101aef36-2dcc-4a46-9926-02c27441476c
begin
	dt2 = StatsBase.fit(ZScoreTransform, X_select; dims=1)
	X_select_std = StatsBase.transform(dt2,X_select)
end;

# ╔═╡ 8aba9a19-8733-408f-bc4f-e58dd415401c
begin
	B = 200;
	ns = length(y[train] );
	betas_bootstrap = zeros(length(mdl.betas),B);
	for b = 1:B
	  # Sample training samples with replacement
	  bootstrap_samples = sample(1:ns,ns,replace=true);
	  betas_bootstrap[:,b] = glmnet(X_select_std[bootstrap_samples,:], 
	    y[bootstrap_samples]; 
	    alpha=0.5, 
	    lambda=[lambdamin(cv)],
	    standardize=false).betas;
	end
end

# ╔═╡ 30e46e18-0560-475f-9946-e539dcb0aa12
lambdamin(cv)

# ╔═╡ 3d8ddba6-9f91-4b9f-8743-ae9072d77556
md"Compute the bootstrap frequencies"

# ╔═╡ 637c83a8-419b-40b4-8327-788cc6a8b51b
freq_bootstrap = vec(sum(betas_bootstrap .!= 0, dims=2)) / B;

# ╔═╡ ec3fd049-7037-4d42-94fd-a5e10a442a73
md"Visualize the bootstrap frequencies and choose an appropriate cutoff by moving the slider."

# ╔═╡ a7398194-f661-42ac-b6d2-0bf21e9c11d1
@bind fcut Slider(0:0.1:1, default=0.6)

# ╔═╡ 574d5df0-4ca4-4ed3-b57a-4d0e868556c3
begin
	histogram(freq_bootstrap, xlabel="Bootstrap frequency", label="")
	vline!([fcut],linewidth=2,label="")
end

# ╔═╡ 6e96c038-e3c7-44eb-81ef-bda3741fab55
md"Print gene names and frequencies above cutoff"

# ╔═╡ 85f4cc21-5e09-4fd0-93a9-e3d436ab8879
begin
	genes_bootstrap = DataFrame("Gene" => names_select[freq_bootstrap .>= fcut], "Frequency" => freq_bootstrap[freq_bootstrap .>= fcut]);
	sort!(genes_bootstrap, :Frequency, rev=true)
end

# ╔═╡ c2559a8e-f3f3-4fa5-aec7-f54ef115bf82
md"
## Figures

Make the same panels as in the CCLE paper figure 2. First make a data matrix of the selected features (genes) for the heatmap, with samples sorted by increasing value of the sensitivity ``y``:
"

# ╔═╡ a1c76f88-2c97-48be-b472-6b9802b3df72
spy = sortperm(y[train]);

# ╔═╡ d0f536c4-6bab-4845-9724-704e1bb740e1
X_hm = Xtrain_select_std[spy,indexin(genes_bootstrap.Gene, names_select)]';

# ╔═╡ bbdbbc61-4a26-47bc-a603-f4aef69913ba
X_hm_mx = min(maximum(X_hm), abs(minimum(X_hm)))

# ╔═╡ bb709f8a-27c4-4371-83f2-7f61d0c473b2
X_hm[X_hm .> X_hm_mx] .= X_hm_mx;

# ╔═╡ f8aeb775-ca83-4dff-859c-168b5ae5a342
X_hm[X_hm .< -X_hm_mx] .= -X_hm_mx;

# ╔═╡ c4a06b6b-544a-4b67-8d97-7ec5d774365e
phm = heatmap(X_hm, c=:balance, xticks=[], yticks=(1:length(genes_bootstrap.Gene), genes_bootstrap.Gene))

# ╔═╡ 831640f6-d028-423e-8bdf-d831b5aa2946
md"The next plot shows how the sensitivity ``y`` changes across the samples (cell lines), with samples ordered by increasing ``y``:"

# ╔═╡ 75f4c955-72d9-4b0e-b0ec-2a20adcf72e7
plot(y[train][spy], label="")

# ╔═╡ aeff7e90-3af6-4cf7-a5e3-e365b1aa96e3
md"Can you make the bar plot showing the regression weights for the selected features (genes)?
"

# ╔═╡ 2c755683-7f93-4bbd-81e3-0c5b96ee7bab
md"
The documentation of [GLMNet](https://github.com/JuliaStats/GLMNet.jl) explains how to make a plot showing how the regression weights change as the penalty parameter ``\lambda`` varies. Can you make such a plot for our model?
"

# ╔═╡ Cell order:
# ╟─0b0c3970-de94-11ed-1198-03fb54d95a60
# ╠═44b0fb80-c13b-490a-a28c-f45fe69ffe97
# ╠═0bd0d1d8-0bdb-4755-b3db-b1885de15ae4
# ╠═c6664290-bc06-48ae-b586-9ea2f8af5d36
# ╟─96844648-bf72-4232-ab8e-3e552c476ab3
# ╠═8e4a6f00-b274-48b4-870e-cf0cba0e20ff
# ╠═fc0d9eb1-ed41-46c6-b5a1-9ff35fa1789d
# ╠═46b51265-37fa-47d1-b494-b4c8564380b8
# ╠═e006f801-89e1-4b3a-8680-6fc458819877
# ╠═71ba55fa-280b-4993-a70b-da0763d0a4c1
# ╟─a391c536-44a9-4a06-9f8f-79bf60275018
# ╠═005ae139-18ac-47a6-a7bd-a84a4e8d72d1
# ╠═fa8c02f1-70f5-42d4-acb7-791cc3cae92d
# ╟─613688c8-2113-43c7-83d9-25f137212ca9
# ╟─8a000026-152e-44f9-a2f9-f6d44cf3250e
# ╠═92a37123-b262-4cd4-80b2-fa75a8d51b29
# ╟─88521c4f-cd8d-484f-a4b6-4e735702fb9d
# ╟─0c1bc2c8-cbc5-4af2-94ba-004dd7272e57
# ╠═3ed5819f-b5f7-413e-a50d-71faa4a9bc0e
# ╠═ae6af865-a992-4aee-a1d4-c4db8127378a
# ╟─f2b1a375-236e-429a-b3f9-64dea50f9a2f
# ╠═39e157a7-432e-485e-8f42-4152077c52a2
# ╟─9d2246c6-8d68-449c-b258-58eb914e46f8
# ╠═826b0d23-bcf7-433c-82f7-428a39a044e0
# ╟─e071ecf4-2441-4bf5-8830-9546428e27fa
# ╟─03aff45f-206d-44ad-a587-1a8113056f85
# ╠═6519f119-9f91-4ca3-b9ec-d9116149cae3
# ╟─caabc7ea-2f25-4a88-9621-133e384a08d0
# ╠═2fe8a865-00c4-431b-a070-916d0d31a070
# ╟─a6a903a9-9b06-436b-a656-36c71a282117
# ╠═35dbc788-430b-4491-a792-a369fb81b0a5
# ╟─f7092f32-b332-4a61-80c4-46c92e33f7c7
# ╠═469000fe-d028-4e2f-93c2-fff727d373bd
# ╠═fff9698d-06a5-4e75-9ada-734806b66370
# ╟─b284165c-e351-4a7a-980a-aa66b676de8b
# ╠═18ec73f8-9286-4f24-91d1-16596332ae9c
# ╠═101aef36-2dcc-4a46-9926-02c27441476c
# ╠═8aba9a19-8733-408f-bc4f-e58dd415401c
# ╠═30e46e18-0560-475f-9946-e539dcb0aa12
# ╟─3d8ddba6-9f91-4b9f-8743-ae9072d77556
# ╠═637c83a8-419b-40b4-8327-788cc6a8b51b
# ╟─ec3fd049-7037-4d42-94fd-a5e10a442a73
# ╠═a7398194-f661-42ac-b6d2-0bf21e9c11d1
# ╠═574d5df0-4ca4-4ed3-b57a-4d0e868556c3
# ╟─6e96c038-e3c7-44eb-81ef-bda3741fab55
# ╠═85f4cc21-5e09-4fd0-93a9-e3d436ab8879
# ╟─c2559a8e-f3f3-4fa5-aec7-f54ef115bf82
# ╠═a1c76f88-2c97-48be-b472-6b9802b3df72
# ╠═d0f536c4-6bab-4845-9724-704e1bb740e1
# ╠═bbdbbc61-4a26-47bc-a603-f4aef69913ba
# ╠═bb709f8a-27c4-4371-83f2-7f61d0c473b2
# ╠═f8aeb775-ca83-4dff-859c-168b5ae5a342
# ╠═c4a06b6b-544a-4b67-8d97-7ec5d774365e
# ╠═831640f6-d028-423e-8bdf-d831b5aa2946
# ╠═75f4c955-72d9-4b0e-b0ec-2a20adcf72e7
# ╟─aeff7e90-3af6-4cf7-a5e3-e365b1aa96e3
# ╠═2c755683-7f93-4bbd-81e3-0c5b96ee7bab
