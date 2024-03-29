using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using CSV
using XLSX
using GLM
using CategoricalArrays



"""
Download and unzip data from [Albert et al (2018)](https://doi.org/10.7554/eLife.35471) from the following links:

- Gene expression levels: https://figshare.com/ndownloader/files/9547738?private_link=83bddc1ddf3f97108ad4
- Experimental covariates: https://figshare.com/ndownloader/files/10139334?private_link=83bddc1ddf3f97108ad4
- Genotypes: https://figshare.com/ndownloader/files/9547741?private_link=83bddc1ddf3f97108ad4
- eQTL results: https://figshare.com/ndownloader/files/10139337?private_link=83bddc1ddf3f97108ad4
- Heritability results: https://figshare.com/ndownloader/files/10139343?private_link=83bddc1ddf3f97108ad4

Save the files in the `data/raw/Yeast_GRN` directory of the project.

Download and unzip GRN data from Yeastract from:

- https://github.com/michoel-lab/FindrCausalNetworkInferenceOnYeast/blob/main/data/input/yeastract/Expr_TF_act_OR_inh_RegulationMatrix_Documented_2020511.csv.gz
- https://github.com/michoel-lab/FindrCausalNetworkInferenceOnYeast/blob/main/data/input/yeastract/onlyDNABinding_RegulationMatrix_Documented_2020511.csv.gz
"""

"""
Process gene expression data

Note that the expression data file is in R format with missing header for the first column. This is solved by circular permutation of the column names to push the gene names in the right place.
"""

fexpr = datadir("raw", "Yeast_GRN", "SI_Data_01_expressionValues.txt");

df_expr = DataFrame(CSV.File(fexpr))
rename!(df_expr,circshift(names(df_expr),1))
rename!(df_expr, names(df_expr)[1] => "segregant")

"""
Process covariates
"""

fcov = datadir("raw", "Yeast_GRN", "SI_Data_02_covariates.xlsx");

df_cov = DataFrame(XLSX.readtable(fcov, "SI_Data_02_covariates", infer_eltypes=true))

"""
Process genotypes

Same procedure as for the expression data.
"""

fgeno = datadir("raw", "Yeast_GRN", "SI_Data_03_genotypes.txt");

df_geno = DataFrame(CSV.File(fgeno,delim="\t"))
rename!(df_geno,circshift(names(df_geno),1))
rename!(df_geno, names(df_geno)[1] => "segregant")

"""
Shorten long segregant names (A01_01-A01-A1-BYxRM_eQTL_10-H6 etc) used in expression data and covariates to match short names (A01_01 etc) used in genotype data. 
"""

df_expr.segregant = [string(split(x,'-')[1]) for x in df_expr.segregant]

df_cov.segregant = [string(split(x,'-')[1]) for x in df_cov.segregant]

"""
Confirm that segregants are listed in the same order in all dataframes and then remove segregant columns.
"""

@assert all(df_expr.segregant .== df_geno.segregant .== df_cov.segregant)

select!(df_expr, Not(:segregant))
select!(df_cov, Not(:segregant))
select!(df_geno, Not(:segregant))

"""
Correct gene expression levels for covariates

Use a dummy dataframe to call the regression function from the GLM package.
"""

df_expr_adj = copy(df_expr)
df = DataFrame("X1"=>categorical(df_cov.batch), "X2"=>df_cov.OD_covariate, "Z"=>df_expr_adj[:,1])

for i=1:ncol(df_expr_adj)
    df.Z = df_expr[:,i]
    fit = lm(@formula(Z ~ X1 + X2),df)
    df_expr_adj[:,i] = residuals(fit)
end

"""
Process eQTL results
"""

feqtl = datadir("raw", "Yeast_GRN", "SI_Data_04_eQTL.xlsx");
df_eqtl = DataFrame(XLSX.readtable(feqtl, "SI_Data_04_eQTL", infer_eltypes=true))

# Keep only cis-eQTLs
df_eqtl = filter(row -> row.cis==true, df_eqtl)

# Add an rsq column for sorting
df_eqtl.rsq = df_eqtl.r.^2

# Sort by gene alphabetically and rsq in descending order
sort!(df_eqtl, [:gene, :rsq], rev=[false, true])

# Keep only the strongest eQTL for each gene
gdf_eqtl = groupby(df_eqtl, :gene)

df_eqtl = combine(gdf_eqtl, [:pmarker, :r, :rsq] =>
               ((p, r, s) -> (pmarker=p[argmax(s)], r=r[argmax(s)])) =>
               AsTable)

"""
Read Ensmbl annotation file
"""
fens =  datadir("raw", "Yeast_GRN", "Ensembl", "cleaned_genes_ensembl111_yeast_step2.txt");
df_ens = DataFrame(CSV.File(fens, header=false, delim='\t'))

# Split column 9 in separate columns, first splitting on space
id_name_array = [split(x) for x in df_ens[:,9]]
id_array = string.([split(id_name_array[k][1], ":")[2] for k in eachindex(id_name_array)])
name_array = string.([split(id_name_array[k][2], "=")[2] for k in eachindex(id_name_array)])

# Create dict from arrays
id2name = Dict(zip(id_array, name_array))

expr_new_names =  [x in keys(id2name) ? id2name[x] : x for x in names(df_expr)];
rename!(df_expr, expr_new_names)

"""
Process GRN data from Yeastract
"""
fGRN_binding = datadir("raw", "Yeast_GRN", "Yeastract","onlyDNABinding_RegulationMatrix_Documented_2020511.csv");
fGRN_expression = datadir("raw", "Yeast_GRN", "Yeastract","Expr_TF_act_OR_inh_RegulationMatrix_Documented_2020511.csv");

df_GRN_binding = DataFrame(CSV.File(fGRN_binding))
df_GRN_expression = DataFrame(CSV.File(fGRN_expression))



               
"""
Save data to processed directory
"""
fexpr_out = datadir("processed", "Yeast_GRN", "Yeast_GRN-expr.csv");
CSV.write(fexpr_out, df_expr)

fexpr_adj_out = datadir("processed", "Yeast_GRN", "Yeast_GRN-expr_adj.csv");
CSV.write(fexpr_adj_out, df_expr_adj)

fgeno_out = datadir("processed", "Yeast_GRN", "Yeast_GRN-geno.csv");
CSV.write(fgeno_out, df_geno)

feqtl_out = datadir("processed", "Yeast_GRN", "Yeast_GRN-eQTL.csv");
CSV.write(feqtl_out, df_eqtl)
