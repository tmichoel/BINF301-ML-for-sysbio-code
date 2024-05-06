### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 73f8fd00-093c-11ef-3d96-6f1f3b3df110
using DrWatson

# ╔═╡ 485e8c6e-e591-4b33-9110-b8c69e19d194
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ ea4a72ec-aaaa-44c9-af49-4ca48dde7d29
begin
	using DataFrames
	using CSV
	using Arrow
	using DataSets
	using BioFindr
	using Statistics
	using StatsBase
	using Distributions
	using StatsPlots, LaTeXStrings
	using PlutoUI
	using Printf
end

# ╔═╡ f0e0d37f-3d70-4060-9499-e6c8955e4c5a
md"#### Expression data"

# ╔═╡ 31279065-2a66-4078-8372-e51e7990ecbd
fexpr = datadir("processed","Yeast_GRN","Yeast_GRN-expr_adj.arrow");

# ╔═╡ ecfd2c4b-b538-41f6-bcc2-40e8bf1d9a13
df_expr = DataFrame(Arrow.Table(fexpr))

# ╔═╡ c373b934-97a5-46a5-90c9-f6c796711aa1
md"#### Genotype data"

# ╔═╡ 4f7a63ca-fa2d-4ee3-8935-de7f10c62d81
fgeno = datadir("processed","Yeast_GRN","Yeast_GRN-geno.arrow");

# ╔═╡ 534af84e-8dcb-4f00-a2fa-7a98b8eb4e2c
df_geno = DataFrame(Arrow.Table(fgeno))

# ╔═╡ afe36cae-5275-4b18-a997-9ee8c6446a55
md"#### eQTL mapping data"

# ╔═╡ 12dd959f-8078-4475-b048-83a90db59eba
feqtl = datadir("processed", "Yeast_GRN", "Yeast_GRN-eQTL.arrow");

# ╔═╡ 6901d470-5bf9-4a25-8086-f86d94468535
df_eqtl = DataFrame(Arrow.Table(feqtl))

# ╔═╡ 9e7907e3-6516-430d-8c06-6cb316522c31
md"#### GRN data"

# ╔═╡ d7540e17-ddd8-4469-9c81-6a07f98f3039
fgrn_binding = datadir("processed", "Yeast_GRN", "Yeast_GRN-GRN_binding.arrow");

# ╔═╡ e34dedef-885e-4bf1-b5fe-541e1ea4bbf4
df_GRN_binding = DataFrame(Arrow.Table(fgrn_binding))

# ╔═╡ b76935d3-b0ef-4ab8-81db-53303cebb68d
fgrn_perturbation = datadir("processed", "Yeast_GRN", "Yeast_GRN-GRN_perturbation.arrow");

# ╔═╡ 9c7a5213-74a5-47db-99c2-15de3055e426
df_GRN_perturbation = DataFrame(Arrow.Table(fgrn_perturbation))

# ╔═╡ 940b8d86-544e-482d-9307-5154e5be64f4
md"""
## Yeast data

We use expression and genotype data from $(nrow(df_expr)) segregants of a yeast cross from [Albert et al. (2018)](https://doi.org/10.7554/eLife.35471), and gene regulatory networks (GRNs) of DNA binding and response to perturbation of transcription factors (TFs) from [Yeastract](http://www.yeastract.com/).

The data used here has been filtered as follows:

- The expression data contains $(ncol(df_expr)) genes which had both expression data and were present in both GRNs
- The genotype data contains $(ncol(df_geno)) SNPs which were the strongest cis-eQTL for a gene; $(nrow(df_eqtl)) genes has such an eQTL.
- The DNA binding and perturbation GRNs contain targets for respectively $(length(unique(df_GRN_binding.Source))) and $(length(unique(df_GRN_perturbation.Source))) TFs which also had expression data available. 
"""

# ╔═╡ 75a51d9b-5bae-4e15-8ecc-0b7aaa07ae71
TF_common = intersect(unique(df_GRN_binding.Source),unique(df_GRN_perturbation.Source),df_eqtl.Gene)

# ╔═╡ a7411227-b669-4b35-9def-a0a96b8a9abb
md"""
## Prepare the analysis

We will reconstruct and evaluate gene regulatory networks for a subset of $(length(TF_common)) TFs which have an eQTL and have targets in both the DNA binding and perturbation GRNs.
"""

# ╔═╡ 6884489e-7f3a-441b-9da7-c90328daa754
md"Filter the eQTL data to keep only eQTLs for the common TFs:"

# ╔═╡ dca0b3c4-0e20-4c17-a706-33df9a047cce
df_eqtl_TF = subset(df_eqtl, :Gene => x ->  in.(x, Ref(TF_common)))

# ╔═╡ c9b24374-85c5-42df-8131-fbd8f5d3aa51
md"Filter the GRNs to keep only targets of the common Ts:"

# ╔═╡ ea378f52-e26a-4db6-8d9a-1d58cab5cefa
df_GRN_binding_sub = subset(df_GRN_binding, :Source => x ->  in.(x, Ref(TF_common)));

# ╔═╡ feab1bcf-8180-48ac-8794-f382ab9a114e
df_GRN_perturbation_sub = subset(df_GRN_perturbation, :Source => x ->  in.(x, Ref(TF_common)));

# ╔═╡ 9bce6489-1bef-4ef3-8dfa-e2b5df7d0201
md"Keep also a GRN that contains only targets that are in both the DNA binding and perturbation GRN:"

# ╔═╡ e74fed5f-6a4c-47bd-a1f2-d19510c2c827
df_GRN_common = innerjoin(df_GRN_binding_sub,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 985daec1-3bbd-478d-b416-6cb1dd4fedf0
md"""
## Network reconstruction

We will reconstruct causal and coexpression-based networks using the [BioFindr](https://github.com/tmichoel/BioFindr.jl) tool. See the `Causal_inference.jl` notebook for a background on the statistical principles of testing for causal effects, and the [BioFindr Tutorials](https://lab.michoel.info/BioFindrTutorials/) for more information about using [BioFindr](https://github.com/tmichoel/BioFindr.jl). 

The analysis presented here is a simplified version of the analysis presented in the paper [Comparison between instrumental variable and mediation-based methods for reconstructing causal gene networks in yeast](https://doi.org/10.1039/D0MO00140F).

We will reconstruct three GRNs:

- A [coexpression-based GRN](https://lab.michoel.info/BioFindrTutorials/coexpression.html), named ``P_0`` in [the paper](https://doi.org/10.1039/D0MO00140F) and `dP_cor` here.
- A [mediation-based causal GRN](https://lab.michoel.info/BioFindrTutorials/causal-inference.html#causal-test-combinations), named ``P_2P_3`` in [the paper](https://doi.org/10.1039/D0MO00140F) and `dP_med` here.
- An [instrumental variable-based causal GRN](https://lab.michoel.info/BioFindrTutorials/causal-inference.html#causal-test-combinations), named ``P_2P_5`` in [the paper](https://doi.org/10.1039/D0MO00140F) and `dP_IV` here.
"""

# ╔═╡ d95651d8-6313-472d-bce0-a335f89a4155
# ╠═╡ show_logs = false
dP_cor = findr(df_expr, colnames=TF_common);

# ╔═╡ 8869c152-44b1-4848-9cb4-89d530aaa4bc
# ╠═╡ show_logs = false
dP_IV = findr(df_expr,df_geno,df_eqtl_TF; colX="Gene", colG="SNP");

# ╔═╡ ccfb70db-12e8-4d40-b692-4f29a545e1e9
dP_med = findr(df_expr,df_geno,df_eqtl_TF; colX="Gene", colG="SNP", combination="mediation");

# ╔═╡ 5f471134-4927-4d84-814e-2bbbea80e20d
md"""
## Network analysis

### Precision vs. recall curves

An important evaluation of networks reconstructed from omics data is to test if their predicted confidence level (the `Probability` columns in `dP_cor`, `dP_IV`, and `dP_med`) is indicative of the experimentally confirmed interactions in the DNA binding or perturbation GRNs.

This is typically done by computing the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) at different confidence levels.
"""

# ╔═╡ 6c4b9c2e-2982-4fe2-8391-8133350e932a
md"We use a simple function to compute precision and recall which takes as input a vector of `labels` (whether an interaction is in the known GRN or not) sorted by decreasing confidence level of the predicted network and the size (number of edges) of the known GRN. Do you understand this function?"

# ╔═╡ 15ef9ddb-1d2a-49d8-9975-663a44ec81c8
function recall_precision(labels,n)
	recall = cumsum(labels) ./ n
	precision = cumsum(labels) ./ vec(1:length(labels))
	return recall, precision
end

# ╔═╡ 512fa917-185b-4864-8703-13527d7f792f
md"To call this function we need to add the GRN labels to the network predictions:"

# ╔═╡ f069136c-a135-417d-8c97-a8314cae86db
leftjoin!(dP_cor,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ a2b3b2da-e555-4ecc-9b9b-f3a116ee418c
leftjoin!(dP_cor,df_GRN_binding_sub, on = [:Source, :Target]);

# ╔═╡ 31cd3c3e-a4d6-4a55-aa22-81c8638830d8
dP_cor.Perturbation[ismissing.(dP_cor.Perturbation)] .= 0;

# ╔═╡ 13b29419-9c03-4009-b3c6-7e2827e46ee1
dP_cor.Binding[ismissing.(dP_cor.Binding)] .= 0;

# ╔═╡ 82528f24-73b2-4b40-a67e-6edc21e1a456
leftjoin!(dP_IV,df_GRN_binding_sub, on = [:Source, :Target]);

# ╔═╡ b39728a5-1c69-492a-af66-1f061c1af53f
leftjoin!(dP_IV,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 8ca83c0c-8155-4714-9ab2-2b5ed95052c5
dP_IV.Perturbation[ismissing.(dP_IV.Perturbation)] .= 0;

# ╔═╡ bf67cf03-d86e-4487-9eec-0ac6a5124419
dP_IV.Binding[ismissing.(dP_IV.Binding)] .= 0;

# ╔═╡ 8c337ad9-b0c3-4cb0-847a-145ac54ce534
leftjoin!(dP_med,df_GRN_binding_sub, on = [:Source, :Target]);

# ╔═╡ 83af7f0a-0135-48b3-900d-d4e439d9caff
leftjoin!(dP_med,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 5a4ebce6-204d-407e-a7a9-a12e71b400b0
dP_med.Perturbation[ismissing.(dP_med.Perturbation)] .= 0;

# ╔═╡ 1303ec2f-b8a8-495c-bd72-bc7f3fe32814
dP_med.Binding[ismissing.(dP_med.Binding)] .= 0;

# ╔═╡ a4b6cb89-d287-433c-9db5-3b3974858796
md"Now compute the recall and precision vectors:"

# ╔═╡ c0d33354-97fe-4609-a176-5c3ea8875955
recall_pert_cor, precision_pert_cor = recall_precision(dP_cor.Perturbation,nrow(df_GRN_perturbation_sub));

# ╔═╡ 53f0ad15-5196-4849-8c9a-f4a017696e58
recall_pert_IV, precision_pert_IV = recall_precision(dP_IV.Perturbation, nrow(df_GRN_perturbation_sub));

# ╔═╡ 0258106d-84f6-4d45-a356-03f4943679d1
recall_pert_med, precision_pert_med = recall_precision(dP_med.Perturbation,nrow(df_GRN_perturbation_sub));

# ╔═╡ 7258c424-1f80-4af9-9df8-76f2a1b58fd6
recall_bind_cor, precision_bind_cor = recall_precision(dP_cor.Binding,nrow(df_GRN_binding_sub));

# ╔═╡ c393a70a-e4a3-47f8-ba4f-b4701fc181f8
recall_bind_IV, precision_bind_IV = recall_precision(dP_IV.Binding,nrow(df_GRN_binding_sub));

# ╔═╡ 9b6d46c0-a51b-4603-b12d-0a9e9168b580
recall_bind_med, precision_bind_med = recall_precision(dP_med.Binding,nrow(df_GRN_binding_sub));

# ╔═╡ 068f3032-331c-4d28-9187-328f45567c5a
begin
	cut_pert = 0.025
	plot(recall_pert_cor[recall_pert_cor.<cut_pert],precision_pert_cor[recall_pert_cor.<cut_pert], label="Coexpression GRN")
	plot!(recall_pert_med[recall_pert_med.<cut_pert],precision_pert_med[recall_pert_med.<cut_pert], label="Causal GRN (mediation)")
	plot!(recall_pert_IV[recall_pert_IV.<cut_pert],precision_pert_IV[recall_pert_IV.<cut_pert], label="Causal GRN (IV)")
	xlabel!("Recall")
	ylabel!("Precision")
	title!("Precision-recall for the perturbation GRN")
end

# ╔═╡ 9aa6e601-d3e0-47bf-b9b9-bfc84d3fd792
begin
	cut_bind = 0.025
	plot(recall_bind_cor[recall_bind_cor.<cut_bind],precision_bind_cor[recall_bind_cor.<cut_bind], label="Coexpression GRN")
	plot!(recall_bind_med[recall_bind_med.<cut_bind],precision_bind_med[recall_bind_med.<cut_bind], label="Causal GRN (mediation)")
	plot!(recall_bind_IV[recall_bind_IV.<cut_bind],precision_bind_IV[recall_bind_IV.<cut_bind], label="Causal GRN (IV)")
	xlabel!("Recall")
	ylabel!("Precision")
	title!("Precision-recall for the DNA binding GRN")
end

# ╔═╡ 53f3f9b1-c170-44bb-82ba-61b9e93ca166
fdcut_IV = 0.1;

# ╔═╡ 8400a795-8c45-4c48-8114-325f7f18244b
md"### Target set enrichment

In the paper [Integrating large-scale functional genomic data to dissect the complexity of yeast regulatory networks](https://doi.org/10.1038/ng.167), target set enrichment was used to evaluate network quality.

For target set enrichment, a cutoff on the predicted confidence level must be set to define a specific set of target for a given TF. Then one tests if the overlap between the predicted and true target set is greater than expected by chance using the [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution). The result of the test is expressed as a [p-value](https://en.wikipedia.org/wiki/P-value). 

Let's do target set enrichment for the causal IV network. We (arbitrarily) define  an [FDR](https://en.wikipedia.org/wiki/False_discovery_rate) cutoff of $(fdcut_IV).
"

# ╔═╡ caa0c224-1014-4254-98a3-d92253832db2
md"First for the perturbation GRN:"

# ╔═╡ b8fa4378-7e4d-42f6-8761-cec753621b55
hg_pval_pert = zeros(size(TF_common));

# ╔═╡ 19d3446d-c767-421d-a112-666f19b40011
for k in eachindex(TF_common)
	# predicted target set
	set_predicted = dP_IV.Target[dP_IV.Source .== TF_common[k] .&& dP_IV.qvalue .<= fdcut_IV]
	# true target set
	set_true = df_GRN_perturbation_sub.Target[df_GRN_perturbation_sub.Source .== TF_common[k]]
	# parameters of the hypergeometric distribution
	s = length(set_true)
	f = nrow(df_GRN_perturbation_sub) - s
	n = length(set_predicted)
	x = length(intersect(set_true,set_predicted))
	# p-value
	hg_pval_pert[k] = ccdf( Hypergeometric(s,f,n), x )
end

# ╔═╡ 2de0a13d-9762-43d5-b8e7-44fef0942409
md"Likewise for the DNA binding GRN:"

# ╔═╡ 0a126189-9c71-4365-9b03-e96a0535ad52
hg_pval_bind = zeros(size(TF_common));

# ╔═╡ fb7cb36d-f45c-47b4-91ce-43d8e2f40d77
for k in eachindex(TF_common)
	# predicted target set
	set_predicted = dP_IV.Target[dP_IV.Source .== TF_common[k] .&& dP_IV.qvalue .<= fdcut_IV]
	# true target set
	set_true = df_GRN_binding_sub.Target[df_GRN_binding_sub.Source .== TF_common[k]]
	# parameters of the hypergeometric distribution
	s = length(set_true)
	f = nrow(df_GRN_binding_sub) - s
	n = length(set_predicted)
	x = length(intersect(set_true,set_predicted))
	# p-value
	hg_pval_bind[k] = ccdf( Hypergeometric(s,f,n), x )
end

# ╔═╡ 54b3bedd-ece6-465e-9e66-f66cf3ad9885
sort!( DataFrame(:TF => TF_common, :pvalue_pert => hg_pval_pert, :pvalue_bind => hg_pval_bind), :pvalue_pert )

# ╔═╡ e9ac5e0e-b069-4f05-858e-9a40526335e9
md"Compare the enrichment p-values:"

# ╔═╡ a47732b0-012b-4090-9e73-51068b66e9bf
begin
	scatter(-log10.(hg_pval_bind), -log10.(hg_pval_pert), label="")
	xlabel!(L"Target enrichment DNA binding ($-\log_{10} p$)")
	ylabel!(L"Target enrichment perturbation ($-\log_{10} p$)")
end

# ╔═╡ Cell order:
# ╠═73f8fd00-093c-11ef-3d96-6f1f3b3df110
# ╠═485e8c6e-e591-4b33-9110-b8c69e19d194
# ╠═ea4a72ec-aaaa-44c9-af49-4ca48dde7d29
# ╟─940b8d86-544e-482d-9307-5154e5be64f4
# ╟─f0e0d37f-3d70-4060-9499-e6c8955e4c5a
# ╠═31279065-2a66-4078-8372-e51e7990ecbd
# ╠═ecfd2c4b-b538-41f6-bcc2-40e8bf1d9a13
# ╟─c373b934-97a5-46a5-90c9-f6c796711aa1
# ╠═4f7a63ca-fa2d-4ee3-8935-de7f10c62d81
# ╠═534af84e-8dcb-4f00-a2fa-7a98b8eb4e2c
# ╟─afe36cae-5275-4b18-a997-9ee8c6446a55
# ╠═12dd959f-8078-4475-b048-83a90db59eba
# ╠═6901d470-5bf9-4a25-8086-f86d94468535
# ╟─9e7907e3-6516-430d-8c06-6cb316522c31
# ╠═d7540e17-ddd8-4469-9c81-6a07f98f3039
# ╠═e34dedef-885e-4bf1-b5fe-541e1ea4bbf4
# ╠═b76935d3-b0ef-4ab8-81db-53303cebb68d
# ╠═9c7a5213-74a5-47db-99c2-15de3055e426
# ╟─a7411227-b669-4b35-9def-a0a96b8a9abb
# ╠═75a51d9b-5bae-4e15-8ecc-0b7aaa07ae71
# ╟─6884489e-7f3a-441b-9da7-c90328daa754
# ╠═dca0b3c4-0e20-4c17-a706-33df9a047cce
# ╟─c9b24374-85c5-42df-8131-fbd8f5d3aa51
# ╠═ea378f52-e26a-4db6-8d9a-1d58cab5cefa
# ╠═feab1bcf-8180-48ac-8794-f382ab9a114e
# ╟─9bce6489-1bef-4ef3-8dfa-e2b5df7d0201
# ╠═e74fed5f-6a4c-47bd-a1f2-d19510c2c827
# ╟─985daec1-3bbd-478d-b416-6cb1dd4fedf0
# ╠═d95651d8-6313-472d-bce0-a335f89a4155
# ╠═8869c152-44b1-4848-9cb4-89d530aaa4bc
# ╠═ccfb70db-12e8-4d40-b692-4f29a545e1e9
# ╟─5f471134-4927-4d84-814e-2bbbea80e20d
# ╟─6c4b9c2e-2982-4fe2-8391-8133350e932a
# ╠═15ef9ddb-1d2a-49d8-9975-663a44ec81c8
# ╟─512fa917-185b-4864-8703-13527d7f792f
# ╠═f069136c-a135-417d-8c97-a8314cae86db
# ╠═a2b3b2da-e555-4ecc-9b9b-f3a116ee418c
# ╠═31cd3c3e-a4d6-4a55-aa22-81c8638830d8
# ╠═13b29419-9c03-4009-b3c6-7e2827e46ee1
# ╠═82528f24-73b2-4b40-a67e-6edc21e1a456
# ╠═b39728a5-1c69-492a-af66-1f061c1af53f
# ╠═8ca83c0c-8155-4714-9ab2-2b5ed95052c5
# ╠═bf67cf03-d86e-4487-9eec-0ac6a5124419
# ╠═8c337ad9-b0c3-4cb0-847a-145ac54ce534
# ╠═83af7f0a-0135-48b3-900d-d4e439d9caff
# ╠═5a4ebce6-204d-407e-a7a9-a12e71b400b0
# ╠═1303ec2f-b8a8-495c-bd72-bc7f3fe32814
# ╟─a4b6cb89-d287-433c-9db5-3b3974858796
# ╠═c0d33354-97fe-4609-a176-5c3ea8875955
# ╠═53f0ad15-5196-4849-8c9a-f4a017696e58
# ╠═0258106d-84f6-4d45-a356-03f4943679d1
# ╠═7258c424-1f80-4af9-9df8-76f2a1b58fd6
# ╠═c393a70a-e4a3-47f8-ba4f-b4701fc181f8
# ╠═9b6d46c0-a51b-4603-b12d-0a9e9168b580
# ╠═068f3032-331c-4d28-9187-328f45567c5a
# ╠═9aa6e601-d3e0-47bf-b9b9-bfc84d3fd792
# ╟─8400a795-8c45-4c48-8114-325f7f18244b
# ╠═53f3f9b1-c170-44bb-82ba-61b9e93ca166
# ╟─caa0c224-1014-4254-98a3-d92253832db2
# ╠═b8fa4378-7e4d-42f6-8761-cec753621b55
# ╠═19d3446d-c767-421d-a112-666f19b40011
# ╟─2de0a13d-9762-43d5-b8e7-44fef0942409
# ╠═0a126189-9c71-4365-9b03-e96a0535ad52
# ╠═fb7cb36d-f45c-47b4-91ce-43d8e2f40d77
# ╠═54b3bedd-ece6-465e-9e66-f66cf3ad9885
# ╟─e9ac5e0e-b069-4f05-858e-9a40526335e9
# ╟─a47732b0-012b-4090-9e73-51068b66e9bf
