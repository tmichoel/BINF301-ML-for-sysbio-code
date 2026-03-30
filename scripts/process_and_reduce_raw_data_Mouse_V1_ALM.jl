using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using Arrow
using CSV
using SparseArrays


"""
Download and unzip gene-level read count zip files from: https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq and store in the folder data/raw/Mouse_V1_ALM

Download a cell cluster annotation file from: https://raw.githubusercontent.com/berenslab/rna-seq-tsne/master/data/tasic-sample_heatmap_plot_data.csv and store in the folder data/processed/Mouse_V1_ALM
"""

fALM_in = datadir("raw","Mouse_V1_ALM","mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_exon-matrix.csv");

fVIS_in = datadir("raw","Mouse_V1_ALM","mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_exon-matrix.csv");

# Read and merge the files in a DataFrame
dfALM_orig = DataFrame(CSV.File(fALM_in));
#dfALM = select(dfALM_orig, 1:5:ncol(dfALM_orig));

dfVIS_orig = DataFrame(CSV.File(fVIS_in));
#dfVIS = select(dfVIS_orig, 1:5:ncol(dfVIS_orig));


df = innerjoin(dfALM_orig,dfVIS_orig,on=:Column1);

# Read cluster label data
fclust = datadir("processed","Mouse_V1_ALM","tasic-sample_heatmap_plot_data.csv");
df_cell_annot = DataFrame(CSV.File(fclust));

# Read gene annotation
fannot_ALM = datadir("raw","Mouse_V1_ALM","mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_genes-rows.csv");
dfannot_ALM = DataFrame(CSV.File(fannot_ALM));

fannot_VIS = datadir("raw","Mouse_V1_ALM","mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_genes-rows.csv");
dfannot_VIS = DataFrame(CSV.File(fannot_VIS));

# Confirm that annotations are the same
all(dfannot_ALM.gene_entrez_id .== dfannot_VIS.gene_entrez_id)
all(string.(dfannot_ALM.gene_entrez_id) .== string.(df.Column1))


# Find and select cells that have a cluster label
tf = .!isnothing.(indexin(names(df),df_cell_annot.sample_name));
select!(df, findall(tf));

# Create cell annotation df for cell subset
df_cell_annot_sub = df_cell_annot[indexin(names(df), df_cell_annot.sample_name), :]
all(names(df) .== df_cell_annot_sub.sample_name)

# Select genes that have non-zero expression in at least nmin=10 cells; non-zero expression is defined as having a value greater than t=32. Values for nmin and t are from Kobak & Berens 2019,
t = 32
nmin = 10
tfg = sum(eachcol(df .>= t)) .>= nmin;
df = df[tfg,:];

df_gene_annot = dfannot_ALM[tfg,:];

# Save CSV file
fout = datadir("processed", "Mouse_V1_ALM_subset", "mouse_ALM_VISp_gene_expression.csv")
CSV.write(fout, df)

fout_cell = datadir("processed", "Mouse_V1_ALM_subset", "tasic-sample_heatmap_plot_data.csv")
CSV.write(fout_cell, df_cell_annot_sub)

fout_gene = datadir("processed", "Mouse_V1_ALM_subset", "mouse_ALM_VISp_gene_annotation.csv")
CSV.write(fout_gene, df_gene_annot)

