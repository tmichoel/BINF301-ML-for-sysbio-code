### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ afaa8b50-e927-11ee-2cdb-0f9d4bca6f43
using DrWatson

# ╔═╡ 51d727e1-4227-46ca-abf7-38d404fc38ae
# ╠═╡ show_logs = false
quickactivate(@__DIR__)

# ╔═╡ e234d22e-ce9f-4635-90f5-1ce181b9dc2d
begin
	using DataFrames
	using CSV
	using DataSets
	using Statistics
	using StatsBase
	using AbstractGPs
	using GLMakie
end

# ╔═╡ d46058b5-1744-4786-9b9b-be56abbcf186
using LinearAlgebra

# ╔═╡ 19a78481-5304-4ce7-ae12-1429264e2b6b
using SpatialOmicsGPs

# ╔═╡ f19e865f-c104-4d77-b0df-57b06627177e
DataSets.load_project!(projectdir("Data.toml"))

# ╔═╡ cae5bfff-831e-4be5-894e-643f7c184f0d
ds = DataSets.open(dataset("Mouse_hypothal_spatial"))

# ╔═╡ 6fe8accc-c3b1-4856-afd1-803e53dd2dfe
# ╠═╡ show_logs = false
df = open(Vector{UInt8}, ds) do buf
           CSV.read(buf, DataFrame);
       end;

# ╔═╡ e250403d-a008-4f6a-89f7-cf321342331f
df_coord=df[:,1:2];

# ╔═╡ dc34a5b0-9708-4377-9ad1-318d15012b58
df_expr=df[:,5:end];

# ╔═╡ d808b610-bf1c-4e14-8e08-1d450aa1a4d2
Y = Matrix(df_expr);

# ╔═╡ 4eb395ca-6835-4a93-b7de-9be21869c5a2
tf = vec(mean(Y .== 0, dims=2)) .< 0.5;

# ╔═╡ cf3729dc-eee3-4e5b-a9d1-019c06df267b
Y[tf,:];

# ╔═╡ ed77dce6-b3da-47b9-ad16-998623413755
begin
	dt = StatsBase.fit(ZScoreTransform, Y[tf,:]; dims=1)
	Y_std = StatsBase.transform(dt,Y[tf,:])
end;

# ╔═╡ 6a59468b-63d7-43c5-bd58-26a6fc61afa7
hist(log2.(Y .+ 1.)[:])

# ╔═╡ f771ce7c-5174-4544-bdf4-f4a50bcccd28
scatter(df_coord.Centroid_X[tf],df_coord.Centroid_Y[tf],color=df_expr[tf,15],colormap = :Blues)

# ╔═╡ 7ea1c7f6-e05f-4b59-bfb1-dd23a70fc113
sd = [std(x) for x in eachcol(df_expr)];

# ╔═╡ b69cb95f-86ae-4b18-b10d-350a148bba5a
argmax(sd)

# ╔═╡ 5b49a03d-ea84-4ee2-89b7-d4b770a87370
x = RowVecs(Matrix(df_coord[tf,:]))[1:end]

# ╔═╡ 3e71250d-78c6-4455-845b-6a59fd9de673
begin
	ls = 100.
	σₛ = 1.0
	σᵣ = 1.0
end;

# ╔═╡ cc1f2ff5-8515-4655-af64-d62c4bc17a8a
kernel =  with_lengthscale(SqExponentialKernel(), ls)

# ╔═╡ 06b02228-b72b-4f69-8e51-fa3e6082e46c
f = GP(kernel)

# ╔═╡ c274a33b-31e1-4031-9b8b-31426f8ad26d
K = kernelmatrix(kernel,x);

# ╔═╡ e33331ac-4247-45a4-8faa-fdef71037e9d
heatmap(K)

# ╔═╡ 82c5a73a-cba4-4b9a-afbc-653958e30ae8
X=Matrix(df_coord[tf,:][1:end,:])

# ╔═╡ 27b0f3f2-8f8a-44da-9566-7550a2317d06
y = df_expr[tf,15][1:end]

# ╔═╡ dffa79c8-a93d-4a74-9365-312cd2f0e656
Z = ones(size(y))

# ╔═╡ f186b076-29ba-4f36-a6df-904621f91b3a
y2,K2 = project_orth_X(y,K,Z)

# ╔═╡ 1664e725-b5a6-4997-b84a-0506a667bacc
K22 = (K2 + K2') / 2;

# ╔═╡ c9f2367e-771d-4cca-80c4-35e006eec0cf
δ = 0.5

# ╔═╡ a7e29f49-8547-4cd1-b28a-6cb69cf28c47
EF=eigen(K22)

# ╔═╡ 84a61d54-2199-4bb8-8ff7-ee5f8e4510ae
yr=EF.vectors' * y2;

# ╔═╡ 5633c5ae-f226-49c6-ab08-7581fea20594
g(δ) = minus_log_like_fullrank(δ, EF.values, yr)

# ╔═╡ 8d57b8c1-4feb-4e14-838b-55c6fc321288
g(δ)

# ╔═╡ 5b1e88f6-36ba-4838-913d-43da58c49d55
δ_vec = 0.1:0.1:20

# ╔═╡ 538b088f-c45c-4a47-b1c9-6e29ce3c9a90
plot(δ_vec,g.(δ_vec))

# ╔═╡ 403ce6fc-ba35-4008-b6a8-818bb0811ca1
n=length(yr)

# ╔═╡ 1d980f76-ed47-4b01-8111-3731eff7389e
δ_min, res = delta_mle_fullrank(EF.values,yr)

# ╔═╡ e81e722e-90f2-4511-8b45-cbd566462bb4
a,b,c = fastlmm_fullrank(y,K; covariates=Z)

# ╔═╡ Cell order:
# ╠═afaa8b50-e927-11ee-2cdb-0f9d4bca6f43
# ╠═51d727e1-4227-46ca-abf7-38d404fc38ae
# ╠═e234d22e-ce9f-4635-90f5-1ce181b9dc2d
# ╠═f19e865f-c104-4d77-b0df-57b06627177e
# ╠═cae5bfff-831e-4be5-894e-643f7c184f0d
# ╠═6fe8accc-c3b1-4856-afd1-803e53dd2dfe
# ╠═e250403d-a008-4f6a-89f7-cf321342331f
# ╠═dc34a5b0-9708-4377-9ad1-318d15012b58
# ╠═d808b610-bf1c-4e14-8e08-1d450aa1a4d2
# ╠═4eb395ca-6835-4a93-b7de-9be21869c5a2
# ╠═cf3729dc-eee3-4e5b-a9d1-019c06df267b
# ╠═ed77dce6-b3da-47b9-ad16-998623413755
# ╠═6a59468b-63d7-43c5-bd58-26a6fc61afa7
# ╠═f771ce7c-5174-4544-bdf4-f4a50bcccd28
# ╠═7ea1c7f6-e05f-4b59-bfb1-dd23a70fc113
# ╠═b69cb95f-86ae-4b18-b10d-350a148bba5a
# ╠═5b49a03d-ea84-4ee2-89b7-d4b770a87370
# ╠═3e71250d-78c6-4455-845b-6a59fd9de673
# ╠═cc1f2ff5-8515-4655-af64-d62c4bc17a8a
# ╠═06b02228-b72b-4f69-8e51-fa3e6082e46c
# ╠═c274a33b-31e1-4031-9b8b-31426f8ad26d
# ╠═e33331ac-4247-45a4-8faa-fdef71037e9d
# ╠═82c5a73a-cba4-4b9a-afbc-653958e30ae8
# ╠═27b0f3f2-8f8a-44da-9566-7550a2317d06
# ╠═dffa79c8-a93d-4a74-9365-312cd2f0e656
# ╠═f186b076-29ba-4f36-a6df-904621f91b3a
# ╠═1664e725-b5a6-4997-b84a-0506a667bacc
# ╠═c9f2367e-771d-4cca-80c4-35e006eec0cf
# ╠═d46058b5-1744-4786-9b9b-be56abbcf186
# ╠═19a78481-5304-4ce7-ae12-1429264e2b6b
# ╠═a7e29f49-8547-4cd1-b28a-6cb69cf28c47
# ╠═84a61d54-2199-4bb8-8ff7-ee5f8e4510ae
# ╠═5633c5ae-f226-49c6-ab08-7581fea20594
# ╠═8d57b8c1-4feb-4e14-838b-55c6fc321288
# ╠═5b1e88f6-36ba-4838-913d-43da58c49d55
# ╠═538b088f-c45c-4a47-b1c9-6e29ce3c9a90
# ╠═403ce6fc-ba35-4008-b6a8-818bb0811ca1
# ╠═1d980f76-ed47-4b01-8111-3731eff7389e
# ╠═e81e722e-90f2-4511-8b45-cbd566462bb4
