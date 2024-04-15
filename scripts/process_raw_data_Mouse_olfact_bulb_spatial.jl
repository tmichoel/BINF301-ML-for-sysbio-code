using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using CSV
using XLSX
using Statistics

"""
Dowload the data manually from the following sources:

https://web.archive.org/web/20160814060152/http://www.spatialtranscriptomicsresearch.org/doi-10-1126science-aaf2403

http://www.spatialomics.org/SpatialDB/st_27365449_details.php

The ST datasets containing the gene counts for each spot are given in two formats, a JSON file that contains one record for each spot-gene count and a count-matrix where gene names are columns and spot coordinates are rows. For all the datasets the spots outside tissue were removed. The JSON files are kept for compatibility with older versions of the ST Viewer (link). The coordinates of the spots were adjusted to account for variations during printing of the arrays.

Use replicate 11 like in the SpatialDE paper.
"""

fname_in = datadir("raw","Mouse_olfact_bulb_spatial","Rep11_MOB_count_matrix-1.tsv")
df = DataFrame(CSV.File(fname_in, delim='\t'));

# Parse the spatial coordinates, stored in the first column
#
# Split the first column into two columns, by splitting on x in df.Column1
df_coord = DataFrame(x = Float64[], y = Float64[])
for i in 1:nrow(df)
    x, y = split(string(df.Column1[i]), "x")
    push!(df_coord, [parse(Float64, x), parse(Float64, y)])
end
# Remove Column1 from df and add the new columns from df_coord as the first two columns
select!(df, Not(:Column1))
insertcols!(df, 1, :x => df_coord.x, :y => df_coord.y)

# Save the processed data
fname_out = datadir("processed","Mouse_olfact_bulb_spatial","Rep11_MOB_count_matrix-1_processed.csv")
CSV.write(fname_out, df);