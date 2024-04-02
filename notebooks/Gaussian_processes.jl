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
x = RowVecs(Matrix(df_coord[tf,:]))[1:20:end]

# ╔═╡ e67e6e37-f66d-4dcc-936d-fa8605c5d8b0
scatter(x.X)

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
km = kernelmatrix(kernel,x);

# ╔═╡ e33331ac-4247-45a4-8faa-fdef71037e9d
heatmap(km)

# ╔═╡ 0c9d3387-a75e-461f-a768-ba0b4defca4d
fx = f(x, σᵣ)

# ╔═╡ 3aadae96-5e3a-4a90-a35c-3d31efc88826
y_sampled = rand(fx)

# ╔═╡ c81982f2-a136-4678-b09c-9cc997a35ba8
hist(y_sampled)

# ╔═╡ 6ccc1b9c-26bb-412d-bf05-b5bde22e80dd
y = Y_std[1:20:end,113]

# ╔═╡ eb203b4f-8062-40fc-9d60-9afd2c7f6792
logpdf(fx,y)

# ╔═╡ 6ccfdf88-f92e-4068-b4ea-792987e9752f
f_posterior = posterior(fx, y)

# ╔═╡ 89239424-ed7f-48cf-a8e4-b60776b91722
logpdf(f_posterior(x), y)

# ╔═╡ 48b03efd-ed9c-4301-b368-385568a29281


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
# ╠═e67e6e37-f66d-4dcc-936d-fa8605c5d8b0
# ╠═3e71250d-78c6-4455-845b-6a59fd9de673
# ╠═cc1f2ff5-8515-4655-af64-d62c4bc17a8a
# ╠═06b02228-b72b-4f69-8e51-fa3e6082e46c
# ╠═c274a33b-31e1-4031-9b8b-31426f8ad26d
# ╠═e33331ac-4247-45a4-8faa-fdef71037e9d
# ╠═0c9d3387-a75e-461f-a768-ba0b4defca4d
# ╠═3aadae96-5e3a-4a90-a35c-3d31efc88826
# ╠═c81982f2-a136-4678-b09c-9cc997a35ba8
# ╠═6ccc1b9c-26bb-412d-bf05-b5bde22e80dd
# ╠═eb203b4f-8062-40fc-9d60-9afd2c7f6792
# ╠═6ccfdf88-f92e-4068-b4ea-792987e9752f
# ╠═89239424-ed7f-48cf-a8e4-b60776b91722
# ╠═48b03efd-ed9c-4301-b368-385568a29281
