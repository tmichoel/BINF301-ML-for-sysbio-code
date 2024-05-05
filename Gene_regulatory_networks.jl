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
	using StatsPlots, LaTeXStrings
	using PlutoUI
	using Printf
end

# ╔═╡ 940b8d86-544e-482d-9307-5154e5be64f4
md"""
## Yeast data
"""

# ╔═╡ 31279065-2a66-4078-8372-e51e7990ecbd
fexpr = datadir("processed","Yeast_GRN","Yeast_GRN-expr_adj.arrow");

# ╔═╡ ecfd2c4b-b538-41f6-bcc2-40e8bf1d9a13
df_expr = DataFrame(Arrow.Table(fexpr))

# ╔═╡ 4f7a63ca-fa2d-4ee3-8935-de7f10c62d81
fgeno = datadir("processed","Yeast_GRN","Yeast_GRN-geno.arrow");

# ╔═╡ 534af84e-8dcb-4f00-a2fa-7a98b8eb4e2c
df_geno = DataFrame(Arrow.Table(fgeno))

# ╔═╡ 12dd959f-8078-4475-b048-83a90db59eba
feqtl = datadir("processed", "Yeast_GRN", "Yeast_GRN-eQTL.arrow");

# ╔═╡ 6901d470-5bf9-4a25-8086-f86d94468535
df_eqtl = DataFrame(Arrow.Table(feqtl))

# ╔═╡ d7540e17-ddd8-4469-9c81-6a07f98f3039
fgrn_binding = datadir("processed", "Yeast_GRN", "Yeast_GRN-GRN_binding.arrow");

# ╔═╡ e34dedef-885e-4bf1-b5fe-541e1ea4bbf4
df_GRN_binding = DataFrame(Arrow.Table(fgrn_binding))

# ╔═╡ b76935d3-b0ef-4ab8-81db-53303cebb68d
fgrn_perturbation = datadir("processed", "Yeast_GRN", "Yeast_GRN-GRN_perturbation.arrow");

# ╔═╡ 9c7a5213-74a5-47db-99c2-15de3055e426
df_GRN_perturbation = DataFrame(Arrow.Table(fgrn_perturbation))

# ╔═╡ 75a51d9b-5bae-4e15-8ecc-0b7aaa07ae71
TF_common = intersect(unique(df_GRN_binding.Source),unique(df_GRN_perturbation.Source),df_eqtl.Gene)

# ╔═╡ dca0b3c4-0e20-4c17-a706-33df9a047cce
df_eqtl_TF = subset(df_eqtl, :Gene => x ->  in.(x, Ref(TF_common)))

# ╔═╡ d95651d8-6313-472d-bce0-a335f89a4155
dP_cor = findr(df_expr, colnames=TF_common);

# ╔═╡ 8869c152-44b1-4848-9cb4-89d530aaa4bc
dP_IV = findr(df_expr,df_geno,df_eqtl_TF; colX="Gene", colG="SNP");

# ╔═╡ ccfb70db-12e8-4d40-b692-4f29a545e1e9
dP_med = findr(df_expr,df_geno,df_eqtl_TF; colX="Gene", colG="SNP", combination="mediation");

# ╔═╡ ea378f52-e26a-4db6-8d9a-1d58cab5cefa
df_GRN_binding_sub = subset(df_GRN_binding, :Source => x ->  in.(x, Ref(TF_common)));

# ╔═╡ feab1bcf-8180-48ac-8794-f382ab9a114e
df_GRN_perturbation_sub = subset(df_GRN_perturbation, :Source => x ->  in.(x, Ref(TF_common)));

# ╔═╡ e74fed5f-6a4c-47bd-a1f2-d19510c2c827
df_GRN_common = innerjoin(df_GRN_binding_sub,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 0d6b39c9-8ccd-47ae-873b-f2642039475d
nrow(df_GRN_common)/nrow(df_GRN_binding_sub)

# ╔═╡ 15ef9ddb-1d2a-49d8-9975-663a44ec81c8
function recall_precision(dP,dGRN)
	recall = cumsum(dP.Perturbation) ./ nrow(dGRN)
	precision = cumsum(dP.Perturbation) ./ vec(1:nrow(dP))
	return recall, precision
end

# ╔═╡ f069136c-a135-417d-8c97-a8314cae86db
leftjoin!(dP_cor,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 31cd3c3e-a4d6-4a55-aa22-81c8638830d8
dP_cor.Perturbation[ismissing.(dP_cor.Perturbation)] .= 0;

# ╔═╡ c0d33354-97fe-4609-a176-5c3ea8875955
recall_cor, precision_cor = recall_precision(dP_cor,df_GRN_perturbation_sub);

# ╔═╡ b39728a5-1c69-492a-af66-1f061c1af53f
leftjoin!(dP_IV,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 8ca83c0c-8155-4714-9ab2-2b5ed95052c5
dP_IV.Perturbation[ismissing.(dP_IV.Perturbation)] .= 0;

# ╔═╡ 53f0ad15-5196-4849-8c9a-f4a017696e58
recall_IV, precision_IV = recall_precision(dP_IV, df_GRN_perturbation_sub);

# ╔═╡ 83af7f0a-0135-48b3-900d-d4e439d9caff
leftjoin!(dP_med,df_GRN_perturbation_sub, on = [:Source, :Target]);

# ╔═╡ 5a4ebce6-204d-407e-a7a9-a12e71b400b0
dP_med.Perturbation[ismissing.(dP_med.Perturbation)] .= 0;

# ╔═╡ 0258106d-84f6-4d45-a356-03f4943679d1
recall_med, precision_med = recall_precision(dP_med,df_GRN_perturbation_sub);

# ╔═╡ 068f3032-331c-4d28-9187-328f45567c5a
begin
	plot(recall_cor[recall_cor.<0.025],precision_cor[recall_cor.<0.025])
	plot!(recall_med[recall_med.<0.025],precision_med[recall_med.<0.025])
	plot!(recall_IV[recall_IV.<0.025],precision_IV[recall_IV.<0.025])
end

# ╔═╡ Cell order:
# ╠═73f8fd00-093c-11ef-3d96-6f1f3b3df110
# ╠═485e8c6e-e591-4b33-9110-b8c69e19d194
# ╠═ea4a72ec-aaaa-44c9-af49-4ca48dde7d29
# ╠═940b8d86-544e-482d-9307-5154e5be64f4
# ╠═31279065-2a66-4078-8372-e51e7990ecbd
# ╠═ecfd2c4b-b538-41f6-bcc2-40e8bf1d9a13
# ╠═4f7a63ca-fa2d-4ee3-8935-de7f10c62d81
# ╠═534af84e-8dcb-4f00-a2fa-7a98b8eb4e2c
# ╠═12dd959f-8078-4475-b048-83a90db59eba
# ╠═6901d470-5bf9-4a25-8086-f86d94468535
# ╠═d7540e17-ddd8-4469-9c81-6a07f98f3039
# ╠═e34dedef-885e-4bf1-b5fe-541e1ea4bbf4
# ╠═b76935d3-b0ef-4ab8-81db-53303cebb68d
# ╠═9c7a5213-74a5-47db-99c2-15de3055e426
# ╠═75a51d9b-5bae-4e15-8ecc-0b7aaa07ae71
# ╠═dca0b3c4-0e20-4c17-a706-33df9a047cce
# ╠═d95651d8-6313-472d-bce0-a335f89a4155
# ╠═8869c152-44b1-4848-9cb4-89d530aaa4bc
# ╠═ccfb70db-12e8-4d40-b692-4f29a545e1e9
# ╠═ea378f52-e26a-4db6-8d9a-1d58cab5cefa
# ╠═feab1bcf-8180-48ac-8794-f382ab9a114e
# ╠═e74fed5f-6a4c-47bd-a1f2-d19510c2c827
# ╠═0d6b39c9-8ccd-47ae-873b-f2642039475d
# ╠═15ef9ddb-1d2a-49d8-9975-663a44ec81c8
# ╠═f069136c-a135-417d-8c97-a8314cae86db
# ╠═31cd3c3e-a4d6-4a55-aa22-81c8638830d8
# ╠═c0d33354-97fe-4609-a176-5c3ea8875955
# ╠═b39728a5-1c69-492a-af66-1f061c1af53f
# ╠═8ca83c0c-8155-4714-9ab2-2b5ed95052c5
# ╠═53f0ad15-5196-4849-8c9a-f4a017696e58
# ╠═83af7f0a-0135-48b3-900d-d4e439d9caff
# ╠═5a4ebce6-204d-407e-a7a9-a12e71b400b0
# ╠═0258106d-84f6-4d45-a356-03f4943679d1
# ╠═068f3032-331c-4d28-9187-328f45567c5a
