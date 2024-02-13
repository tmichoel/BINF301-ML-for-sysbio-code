using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using CSV
using XLSX

"""
Dowload the data manually from the following sources:

- CCLE expression data from [GSE36139](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36139); use the Series Matrix File GSE36139-GPL15308_series_matrix.txt.

- Drug sensitivities: Supplementary Table 11 from the [original publication](https://www.nature.com/articles/nature11003)
"""

"""
EXPRESSION DATA

A series matrix file consists of metadata (all lines beginning with "!") and data (all lines between "!series_matrix_table_begin" and "!series_matrix_table_end").

From the metadata, we need the mapping from GEO sample names (GSM...) to cell line names. This information is in the lines "!Sample_title" and "!Sample_geo_accession". Remove the metadata lines manually and make the sample_title line the header line, and save the result to a new file:
"""
fexpr = datadir("raw","CCLE","GSE36139-GPL15308_series_matrix_nometa.txt");

"""
The following lines do the following:

- Read the data into a dataframe. 
 - Drop genes with missing data. 
 - Extract gene IDs. 
 - Transpose to have genes as columns, samples as rows.
 - Rename the sample column to Cell_line
"""

df = DataFrame(CSV.File(fexpr));
dropmissing!(df);
df = permutedims(df,1);
rename!(df,map(x -> x[1: findfirst('_',x)-1], names(df)))
rename!(df, :Sample => :Cell_line);

"""
Save Entrez gene ids to file for id conversion
"""
entrez_id = DataFrame(ENTREZID = names(df)[2:end]);
CSV.write(datadir("raw","CCLE","CCLE-entrez.csv"),entrez_id);

"""
Read ID conversion file from SynGO
"""
fentrez = datadir("raw","CCLE","SynGO_id_convert_2024-02-13","idmap.csv");
df_entrez = DataFrame(CSV.File(fentrez));
df_entrez.query = string.(df_entrez.query);
rename!(df_entrez, :query => :ENTREZID);

leftjoin!(entrez_id, df_entrez, on=:ENTREZID);
tf = ismissing.(entrez_id.symbol);
entrez_id.symbol[tf] .= entrez_id.ENTREZID[tf];
entrez_id.symbol = string.(entrez_id.symbol);

rename!(df, ["Cell line"; entrez_id.symbol]);

"""
DRUG SENSITIVITY DATA

Convert the drug sensitivity file manually to XLSX format and save the result to a new file:
"""

fsens = datadir("raw","CCLE","41586_2012_BFnature11003_MOESM90_ESM.xlsx");

"""
The following lines do the following:

- Read the data into a dataframe 
- Keep only the column ActArea as a sensitivity measure
- Unstack the dataframe to have cell lines as rows and compounds as columns
- Replace "NA" and missing ActArea values with NaN and convert values to Float64  
- Convert the cell line column to a vector of strings
"""

df_sens = DataFrame(XLSX.readtable(fsens,"Table S11"; first_row=3));
select!(df_sens, :"Primary Cell Line Name" => "Cell_line", :Compound, :ActArea);

df_sens = unstack( df_sens, :Cell_line, :Compound, :ActArea);

for col in eachcol(df_sens)
    replace!(col, "NA" => NaN)
    replace!(col, missing => NaN)
end

for col = 2:ncol(df_sens)
    df_sens[:,col] = convert(Vector{Float64}, df_sens[:,col])
end

df_sens.Cell_line[typeof.(df_sens.Cell_line) .!= String] = string.(df_sens.Cell_line[typeof.(df_sens.Cell_line) .!= String]);
df_sens.Cell_line = convert(Vector{String}, df_sens.Cell_line);

"""
MERGE EXPRESSION AND SENSITIVITY DATA

First merge the expression and sensitivity data on the Cell_lin column using the `inner` join method, then split the merged data again into separate expression and sensitivity dataframes which will now have samples aligned.
"""
df_join = innerjoin(df_sens,df, on=:Cell_line);

dat_sens = select(df_join,2:ncol(df_sens));
dat_expr = select(df_join, Not(1:ncol(df_sens)));

"""
SAVE DATA
"""

fexpr_out = datadir("processed","CCLE","CCLE-expr.csv");
CSV.write(fexpr_out,dat_expr);

fsens_out = datadir("processed","CCLE","CCLE-ActArea.csv");
CSV.write(fsens_out,dat_sens);