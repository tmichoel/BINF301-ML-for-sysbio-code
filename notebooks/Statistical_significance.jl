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

# ╔═╡ 16a821c6-34a6-4ad2-9e89-159cebddf7d6
using DrWatson

# ╔═╡ badcc813-66d6-49c0-bc36-c0f659304bd4
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ 7e672ab8-7208-406a-8ed1-81ee22a2110d
begin
	using DataFrames
	using CSV
	using DataSets
	using Statistics
	using StatsBase
	using StatsPlots, LaTeXStrings
	using FreqTables
	using PlutoUI
	using Printf
	using Random
	using HypothesisTests
	using SmoothingSplines
end

# ╔═╡ 37341592-c0e8-11ee-2da8-87d81712f94f
md"# Statistical significance for genomewide studies
## Setup the environment

Activate the BINF301 environment:
"

# ╔═╡ e2202af3-670e-4d6f-89c8-26304f0e3ad5
md"Load packages:"

# ╔═╡ da338647-0466-4848-a787-f8d6cc6ffdcf
md"
## Load the data

Make sure that you have downloaded the data before running this notebook by executing the script `download_processed_data.jl` in the `scripts` folder. Information to load the data is stored in the `Data.toml` file, which needs to be loaded first:
"

# ╔═╡ 5ed52ecf-d783-4021-af72-1f09d9df1649
DataSets.load_project!(projectdir("Data.toml"))

# ╔═╡ 73199f31-37e5-4eb7-af39-3c1e048f0eb9
md"""
Open the dataset and read each of the files in a DataFrame. This code is adapted from the [DataSets tutorial on JuliaHub](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/).
"""

# ╔═╡ 1e06ac77-62c6-45ca-8e5c-888d31dd2a11
tree = DataSets.open(dataset("TCGA_BRCA"));

# ╔═╡ 1401e3be-18a4-4baf-8ae2-9357d9fbb674
md"
The first file contains clinical data for each sample:
"

# ╔═╡ 8c02f6ca-ff70-4134-b23e-236af499a43d
df_clin = open(Vector{UInt8}, tree["TCGA-BRCA-exp-348-clin.csv"]) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ f0c178b3-e1c0-49ae-927d-88bd0656a21e
md"The second file contains gene expression data, with samples (rows) in the same order as in the clinical data:"

# ╔═╡ f98b3f5c-511c-4057-8b85-ca1d7b8b104a
# ╠═╡ show_logs = false
df_expr = open(Vector{UInt8}, tree["TCGA-BRCA-exp-348-expr.csv"]) do buf
           CSV.read(buf, DataFrame)
       end;

# ╔═╡ 5847f0c6-266d-4e01-ae7c-3ee2dc7b8c5c
ns,ng = size(df_expr);

# ╔═╡ 55052938-f038-487a-a9cd-39b492186f2a
md"
## Create clinical outcome groupings

- Triple negative tumours (yes/no)
- Stage
"

# ╔═╡ 2d6a9c9f-f1a2-4aa7-aba3-fcd1cce6236b
triple_neg = df_clin.:"ER Status" .== "Negative" .&& df_clin.:"PR Status" .== "Negative" .&& df_clin.:"HER2 Final Status" .== "Negative";

# ╔═╡ 09ee5cf8-2dd4-4c3a-93e9-7904e568de32
begin
	stage = zeros(Int16,nrow(df_clin))
	stage[map(x -> ∈(x, ["Stage I", "Stage IA", "Stage IB"]), 
		df_clin.:"AJCC Stage")] .= 1
	stage[map(x -> ∈(x, ["Stage II", "Stage IIA", "Stage IIB"]), 
		df_clin.:"AJCC Stage")] .= 2
	stage[map(x -> ∈(x, ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"]), 
		df_clin.:"AJCC Stage")] .= 3
	stage[df_clin.:"AJCC Stage" .== "Stage IV"] .= 4
end;

# ╔═╡ aaf88a83-05d8-4389-8004-ed31366030ac
md"
## Differential expression analysis for triple negative tumours

There are $(ng) genes and $(ns) samples in the data. There are $(sum(triple_neg)) triple negative and $(sum(.!triple_neg)) non-triple negative tumours. We wish to identify genes that are differentially expressed between triple negative and non-triple negative tumours.
"

# ╔═╡ 19aee9cd-0fb0-4833-93aa-fc342da49b7b
md"
### Subsample and balance the data (optional)

Check the box below to work with a balanced dataset (equal number of triple negative and non-triple negative). If unchecked the same proportion as in the original data is kept:
"

# ╔═╡ c030ed03-ef3e-450f-984b-3f22f14967b5
@bind balanced CheckBox(default=true)

# ╔═╡ e8bc7fe6-b501-44b1-afa2-754acd7d16db
md"
Get the number and indices of triple negative tumours. For a balanced dataset, randomly select the same number from the non-triple negative tumours:
"

# ╔═╡ e8b37b3a-5b4d-4b7f-9d11-74e82589f397
begin
	n_neg = sum(triple_neg);
	idx_neg = findall(triple_neg);
	idx_pos = findall(.!triple_neg);
	if balanced
		idx_pos = sort(shuffle(idx_pos)[1:n_neg])
	end
	n_pos = length(idx_pos);
end;

# ╔═╡ 210b9202-afe0-430b-ad4a-3b1984e25b9e
md"
Move the following slider to subsample a smaller dataset to see the influence of sample size on the downstream analysis. The amount of blue fill is the proportion of tumours that will be sampled in each group.
"

# ╔═╡ 48afe7a6-f5d1-49c7-90c5-13b2446100d2
@bind prop Slider(0.1:0.1:1.0, default=100)

# ╔═╡ 74c18d93-02e7-4b1d-903a-bddfbaf51ed6
if prop < 1.0
	n_neg_select = round(Int,prop * n_neg)
	n_pos_select = round(Int,prop * n_pos)
	idx_neg_select = sort(shuffle(idx_neg)[1:n_neg_select])
	idx_pos_select = sort(shuffle(idx_pos)[1:n_pos_select])
else
	n_neg_select = n_neg
	n_pos_select = n_pos
	idx_neg_select = idx_neg
	idx_pos_select = idx_pos
end;

# ╔═╡ 9853d9c2-a16d-423e-8398-20350ee01581
md"

Make a final dataframe and triple negative selector for the $(n_neg_select) triple negative $(n_pos_select) non-triple negative  tumours sampled for downstream analysis:
"

# ╔═╡ 91532840-1d0d-4366-98a3-46b8b7b4574e
begin
	dfg = [df_expr[idx_neg_select, :]; df_expr[idx_pos_select,:]];
	trip_neg = [triple_neg[idx_neg_select]; triple_neg[idx_pos_select]]
end;

# ╔═╡ 64d4e914-91ca-4102-9657-d76a8a94bc27
md"
### Compute theoretical and empirical p-values

We compute theoretical differential expression p-values using a t-test for each gene:
"

# ╔═╡ abdb9cc7-c20a-415a-9362-a9b87cb3adb7
pₜ = map(x -> pvalue(UnequalVarianceTTest(x[trip_neg], x[.!trip_neg])), eachcol(dfg));

# ╔═╡ 39cc70c6-e29d-44b4-aff5-83b2f11b753e
md"
We compute empirical p-values from the t-test statistics using permuted data. First compute the t-statistics:
"

# ╔═╡ 55b079d1-5e3e-4d6f-9d97-421b51511358
t =  map(x -> UnequalVarianceTTest(x[trip_neg], x[.!trip_neg]).t, eachcol(dfg));

# ╔═╡ afae2e99-ba13-479a-9a69-afa172a840e5
@bind B Slider(20:20:200, default=100)

# ╔═╡ 9c1ce5db-0f3d-4800-a4b7-ef4f5e988f88
md"
Now compute the empirical p-values. Move the slider to select the number of random permutations. B = $(B) selected
"

# ╔═╡ f86e3d22-2bf7-4e97-97ce-96a741629305
# ╠═╡ disabled = true
#=╠═╡
begin
	trand = zeros(ng,B);
	for b = 1:B
	    tf = shuffle(trip_neg);
	    trand[:,b] = map(x -> abs(UnequalVarianceTTest(x[tf], x[.!tf]).t), eachcol(dfg));
	end
end
  ╠═╡ =#

# ╔═╡ 2f582056-41b2-4c04-b482-d924c447d795
# ╠═╡ disabled = true
#=╠═╡
pₑ = map(x -> sum(trand .>= x)./(ng*B), abs.(t));
  ╠═╡ =#

# ╔═╡ 04e1ea77-a1e9-49f8-b11a-f76ba0f4f406
md"Compare empirical and theoretical p-values. In this case we see that they coincide perfectly:"

# ╔═╡ 313226f4-dd3e-4a57-92e2-cd543ef93eb9
# ╠═╡ disabled = true
#=╠═╡
scatter(pₜ,pₑ,label="",xlabel="Theoretical p-values",ylabel="Empirical p-values")
  ╠═╡ =#

# ╔═╡ 003afa7f-612b-4971-bc03-ebc4201d2a4c
md"
Because the theoretical and empirical p-values agree, we will work with the theoretical ones, as they don't have any values identically zero. To play with the sample size slider, you can disable the empirical p-value cells so recomputing the other figures goes faster.
"

# ╔═╡ 482035c4-88ad-4253-8716-60b6a0805556
md"### Estimate $\pi_0$

 To implement the method from Storey & Tibshirani, we First get an estimate of $\pi_0$ for a range of $\lambda$ values.
"


# ╔═╡ 705293b6-4c4f-4593-a116-dad48822bf31
λ = 0.01:0.01:0.95;

# ╔═╡ 5b44462d-564a-4bf5-8c7d-1aa8f757ddd3
π₀_vec = map(x -> sum(pₜ.>x)/(ng*(1-x)),λ);

# ╔═╡ 93b0d03d-dcca-4117-8a79-d0378fd9e26e
md"Now do the cubic spline smoothing. Move the slider to change the smoothness of the spline."

# ╔═╡ dc3d68e1-4d64-48ae-a224-7ca12baabda4
@bind sm Slider(0:0.005:0.02, default=0.01)

# ╔═╡ 2dbc78f3-3fcf-4636-bdc7-f15f788d8045
spl = fit(SmoothingSpline, λ, π₀_vec, sm);

# ╔═╡ f47819cc-3b76-4a72-aaf0-e3b8e1a0d280
md"Extrapolate to get the final estimate:"

# ╔═╡ 4919ecf9-1dd4-497b-9624-24579e1a241b
π₀ = predict(spl,1.0);

# ╔═╡ daa50ae7-b403-4cc8-92f0-bdf64d940ce1
s0 = @sprintf("%1.2f", π₀);

# ╔═╡ 31af70c7-2c81-4f16-8f17-e5a15bc067fd
md"#### Reproduce Fig. 3."

# ╔═╡ 10ed4d29-ae71-4ae4-850e-1b567336281b
begin
	scatter(λ,π₀_vec,label="",xlabel=L"\lambda", ylabel=L"\pi_0(\lambda)")
	plot!(λ,predict(spl,λ),linewidth=3, label="Cubic smoothing spline")
end

# ╔═╡ 2e2561b6-a907-49f2-8d7f-019f376c4c65
md"The spline interpolation curve (red line) depends on a smoothing parameter which determines how closely the line follows the data points being fitted. Go back up and move the slider to see the effect of changing this parameter."


# ╔═╡ e0dfb610-59d1-4ffb-bbf4-91851797efb3
md"#### Reproduce Fig. 1. 

The inset zooms in on the histogram of p-values greater than 0.5, which mostly consist of null p-values. The line shows the estimate of the proportion of null p-values ($\pi_0$=$(s0)). 

Go back and change the sample size. When all available triple negative tumours are included, we find an usually low value for $\pi_0$ for this type of analysis. Due to the very strong difference in global gene expression between triple negative and non-triple negative breast tumours, and the relatively large number of samples for a differential expression analysis we estimate that the majority of genes are truely differentially expressed. On the other hand, when you decrease the sample size, you will see an increase in the estimated proportion of null p-values."


# ╔═╡ 46e444c9-aac0-4ddf-aeb6-7b45da2fcd87
begin
	histogram(pₜ, label="", xlabel="Theoretical p-values", normalize=:pdf)
	histogram!(
	    pₜ,
	    xlims=[0.5, 1],
	    ylims = [0,1.01],
	    label="",
	    normalize=:pdf,
	    inset = (1, bbox(0.05, 0.05, 0.5, 0.5, :top, :right)),
	    subplot = 2,
	    bg_inside = nothing
	)
	hline!([π₀], label="", linewidth=2, subplot=2)
end

# ╔═╡ e753a622-1849-417a-bbb8-4507e73fd919
md"### Estimate q-values
First order the p-values:
"

# ╔═╡ 919e2d49-9980-4e2c-bcb5-7b449dae36a0
begin
	per = sortperm(pₜ);
	pₒ = pₜ[per];
end;

# ╔═╡ 2a26214a-6ef3-4ed9-a223-c41d1823345a
md"Now compute the q-values using the formulae from Storey & Tibshirani:"

# ╔═╡ 3c1ac307-ceb9-4d25-b20a-03747f499a82
begin
	qₒ = zeros(size(pₒ));
	qₒ[end] = π₀ * pₒ[end];
	for k=ng-1:-1:1
	    qₒ[k] = min(π₀*ng*pₒ[k]/k, qₒ[k+1])
	end
end

# ╔═╡ 8765ec48-0366-46a4-9d85-76e188ad3e43
nsig = map(x -> sum(qₒ.<=x),qₒ);

# ╔═╡ c962812a-b47d-4eea-ab59-8e80fc327b2b
nfp = qₒ .* nsig;

# ╔═╡ a239eb85-9517-4ad2-96e1-9822de8b2834
md"#### Reproduce Fig. 2."

# ╔═╡ f8952103-dd33-4296-bf0a-409c8195d21f
begin
	l = @layout [a b;c d];
	p1 = scatter(
	    t[per],qₒ, 
	    label="", xlabel="t-statistics", ylabel="q-values",
	    markersize=.5,  markercolor=:blue,  markerstrokewidth=0)
	
	p2 = scatter(
	    pₒ,qₒ, 
	    label="", xlabel="p-values", ylabel="q-values",
	    markersize=.5,  markercolor=:blue,  markerstrokewidth=0)
	
	p3 = scatter(
	    qₒ, nsig, 
	    label="", xlabel="q-values", ylabel="number of significant genes",
	    markersize=.5,  markercolor=:blue,  markerstrokewidth=0)
	
	p4 = scatter(
	    nsig, nfp, 
	    label="", xlabel="number of significant genes", ylabel="number of exp. FP",
	    markersize=.5,  markercolor=:blue,  markerstrokewidth=0)
	
	plot(p1, p2, p3, p4, layout=l)
end

# ╔═╡ Cell order:
# ╠═37341592-c0e8-11ee-2da8-87d81712f94f
# ╠═16a821c6-34a6-4ad2-9e89-159cebddf7d6
# ╠═badcc813-66d6-49c0-bc36-c0f659304bd4
# ╠═e2202af3-670e-4d6f-89c8-26304f0e3ad5
# ╠═7e672ab8-7208-406a-8ed1-81ee22a2110d
# ╠═da338647-0466-4848-a787-f8d6cc6ffdcf
# ╠═5ed52ecf-d783-4021-af72-1f09d9df1649
# ╟─73199f31-37e5-4eb7-af39-3c1e048f0eb9
# ╠═1e06ac77-62c6-45ca-8e5c-888d31dd2a11
# ╟─1401e3be-18a4-4baf-8ae2-9357d9fbb674
# ╠═8c02f6ca-ff70-4134-b23e-236af499a43d
# ╠═f0c178b3-e1c0-49ae-927d-88bd0656a21e
# ╠═f98b3f5c-511c-4057-8b85-ca1d7b8b104a
# ╠═5847f0c6-266d-4e01-ae7c-3ee2dc7b8c5c
# ╟─55052938-f038-487a-a9cd-39b492186f2a
# ╠═2d6a9c9f-f1a2-4aa7-aba3-fcd1cce6236b
# ╠═09ee5cf8-2dd4-4c3a-93e9-7904e568de32
# ╟─aaf88a83-05d8-4389-8004-ed31366030ac
# ╟─19aee9cd-0fb0-4833-93aa-fc342da49b7b
# ╟─c030ed03-ef3e-450f-984b-3f22f14967b5
# ╟─e8bc7fe6-b501-44b1-afa2-754acd7d16db
# ╠═e8b37b3a-5b4d-4b7f-9d11-74e82589f397
# ╟─210b9202-afe0-430b-ad4a-3b1984e25b9e
# ╟─48afe7a6-f5d1-49c7-90c5-13b2446100d2
# ╠═74c18d93-02e7-4b1d-903a-bddfbaf51ed6
# ╟─9853d9c2-a16d-423e-8398-20350ee01581
# ╠═91532840-1d0d-4366-98a3-46b8b7b4574e
# ╟─64d4e914-91ca-4102-9657-d76a8a94bc27
# ╠═abdb9cc7-c20a-415a-9362-a9b87cb3adb7
# ╟─39cc70c6-e29d-44b4-aff5-83b2f11b753e
# ╠═55b079d1-5e3e-4d6f-9d97-421b51511358
# ╟─9c1ce5db-0f3d-4800-a4b7-ef4f5e988f88
# ╟─afae2e99-ba13-479a-9a69-afa172a840e5
# ╠═f86e3d22-2bf7-4e97-97ce-96a741629305
# ╠═2f582056-41b2-4c04-b482-d924c447d795
# ╟─04e1ea77-a1e9-49f8-b11a-f76ba0f4f406
# ╠═313226f4-dd3e-4a57-92e2-cd543ef93eb9
# ╟─003afa7f-612b-4971-bc03-ebc4201d2a4c
# ╟─482035c4-88ad-4253-8716-60b6a0805556
# ╠═705293b6-4c4f-4593-a116-dad48822bf31
# ╠═5b44462d-564a-4bf5-8c7d-1aa8f757ddd3
# ╟─93b0d03d-dcca-4117-8a79-d0378fd9e26e
# ╠═dc3d68e1-4d64-48ae-a224-7ca12baabda4
# ╠═2dbc78f3-3fcf-4636-bdc7-f15f788d8045
# ╟─f47819cc-3b76-4a72-aaf0-e3b8e1a0d280
# ╠═4919ecf9-1dd4-497b-9624-24579e1a241b
# ╠═daa50ae7-b403-4cc8-92f0-bdf64d940ce1
# ╟─31af70c7-2c81-4f16-8f17-e5a15bc067fd
# ╠═10ed4d29-ae71-4ae4-850e-1b567336281b
# ╟─2e2561b6-a907-49f2-8d7f-019f376c4c65
# ╟─e0dfb610-59d1-4ffb-bbf4-91851797efb3
# ╠═46e444c9-aac0-4ddf-aeb6-7b45da2fcd87
# ╟─e753a622-1849-417a-bbb8-4507e73fd919
# ╠═919e2d49-9980-4e2c-bcb5-7b449dae36a0
# ╟─2a26214a-6ef3-4ed9-a223-c41d1823345a
# ╠═3c1ac307-ceb9-4d25-b20a-03747f499a82
# ╠═8765ec48-0366-46a4-9d85-76e188ad3e43
# ╠═c962812a-b47d-4eea-ab59-8e80fc327b2b
# ╟─a239eb85-9517-4ad2-96e1-9822de8b2834
# ╠═f8952103-dd33-4296-bf0a-409c8195d21f
