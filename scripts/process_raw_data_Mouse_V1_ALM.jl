using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using Arrow
using CSV

"""
Download and unzip gene-level read count zip files from: https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq and store in the folder data/raw/Mouse_V1_ALM

Download a cell cluster annotation file from: https://raw.githubusercontent.com/berenslab/rna-seq-tsne/master/data/tasic-sample_heatmap_plot_data.csv and store in the folder data/processed/Mouse_V1_ALM
"""

# Convert scRNA csv files to arrow format

fALM_in = datadir("raw","Mouse_V1_ALM","mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_exon-matrix.csv");
fALM_out = datadir("raw","Mouse_V1_ALM", "mouse_ALM_2018-06-14_exon-matrix.arrow");

if !isfile(fALM_out)
	Arrow.write(fALM_out, CSV.File(fALM_in));
end

fVIS_in = datadir("raw","Mouse_V1_ALM","mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_exon-matrix.csv");
fVIS_out = datadir("raw","Mouse_V1_ALM", "mouse_VISp_2018-06-14_exon-matrix.arrow");

if !isfile(fVIS_out)
	Arrow.write(fVIS_out, CSV.File(fVIS_in));
end

# Read and merge the arrow files in a DataFrame
dfALM = DataFrame(Arrow.Table(fALM_out));
dfVIS = DataFrame(Arrow.Table(fVIS_out));

df = innerjoin(dfALM,dfVIS,on=:Column1);

# Read cluster label data
fclust = datadir("processed","Mouse_V1_ALM","tasic-sample_heatmap_plot_data.csv");
dfClust = DataFrame(CSV.File(fclust));

# Read gene annotation
fannot_ALM = datadir("raw","Mouse_V1_ALM","mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_genes-rows.csv");
dfannot_ALM = DataFrame(CSV.File(fannot_ALM));

fannot_VIS = datadir("raw","Mouse_V1_ALM","mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_genes-rows.csv");
dfannot_VIS = DataFrame(CSV.File(fannot_VIS));

# Confirm that annotations are the same
all(dfannot_ALM.gene_entrez_id .== dfannot_VIS.gene_entrez_id)
all(string.(dfannot_ALM.gene_entrez_id) .== string.(df.Column1))


# Find and select cells that have a cluster label
tf = .!isnothing.(indexin(names(df),dfClust.sample_name));
select!(df, findall(tf));
#df.Column1 = string.(df.Column1)

# Select genes that have non-zero expression in at least nmin=10 cells; non-zero expression is defined as having a value greater than t=32. Values for nmin and t are from Kobak & Berens 2019,
t = 32
nmin = 10
tfg = sum(eachcol(df .>= t)) .>= nmin;
df = df[tfg,:];

dfannot = dfannot_ALM[tfg,:];

# Save data
fexpr = datadir("processed","Mouse_V1_ALM","mouse_ALM_VISp_gene_expression.arrow");
fannot = datadir("processed","Mouse_V1_ALM","mouse_ALM_VISp_gene_annotation.csv");

Arrow.write(fexpr,df);
CSV.write(fannot,dfannot);


