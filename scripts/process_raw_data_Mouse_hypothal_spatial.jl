using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using CSV
using XLSX
using Statistics

"""
Dowload the data manually from the following sources:

MERFISH data from Supplementary Table 7 from the paper [Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region](https://doi.org/10.1126/science.aau5324)
"""

fexpr = datadir("raw","Mouse_hypothal_spatial","aau5324_moffitt_table-s7.xlsx");

df = DataFrame(XLSX.readtable(fexpr,"Moffitt and Bambah-Mukku et al_"; first_row=2));

# Remove columns that we don't need or have unique values
select!(df, Not(:Cell_ID));
select!(df, Not(:Animal_ID));
select!(df, Not(:Animal_sex));
select!(df, Not(:Behavior));
select!(df, Not(:Bregma));

# Center the X and Y position
df.Centroid_X = df.Centroid_X .- mean(df.Centroid_X);
df.Centroid_Y = df.Centroid_Y .- mean(df.Centroid_Y);

# Convert gene expression columns to Float64
for col in names(df)[5:end]
    df[!, col] = Float64.(df[!, col]);
end

# Save the processed data
fexpr_out = datadir("processed","Mouse_hypothal_spatial","mouse_hypothal_spatial_gene_expression.csv");
CSV.write(fexpr_out,df);