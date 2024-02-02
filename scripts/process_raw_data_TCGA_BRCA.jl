using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using CSV
using XLSX

"""
Dowload the data manually from the following sources:

- BRCA expression data available at https://gdc.cancer.gov/about-data/publications/brca_2012, file BRCA.exp.348.med.txt.
- ER status and cancer stage available in Supplementary Table 1 of the paper at https://www.nature.com/articles/nature11412

Convert the clinical data file to XLSX format and rename and move the files to the following locations, such that the following paths are valid:
"""

fexpr = datadir("raw","TCGA-BRCA","TCGA-BRCA-exp-348-med.txt");
fclin = datadir("raw","TCGA-BRCA","TCGA-BRCA-Supplementary-Tables-1-4.xlsx");

"""
EXPRESSION DATA

Read the data first into a dataframe and fix the following:

- Reduce sample IDs to  12 chars to match the IDs in the clinical data file
- Remove genes with missing data 
"""

df = DataFrame(CSV.File(fexpr));
dropmissing!(df); # drop genes with missing values
df = permutedims(df,1);

# Rename samples
for i=1:lastindex(df[:,1])
    df[i,1] = df[i,1][1:12]
end

"""
CLINICAL DATA

Read the data into a dataframe and rename the sample ID column to match the expression data:
"""

df_clin = DataFrame(XLSX.readtable(fclin,"SuppTable1"; first_row=2));
rename!(df_clin, :"Complete TCGA ID" => :NAME);

"""
MERGE EXPRESSION AND CLINICAL DATA

First merge the expression and clinical data on the sample ID column using the `inner` join method, then split the merged data again into separate expression and clinical dataframes which will now have samples aligned.
"""
df_join = innerjoin(df_clin,df, on=:NAME);

dat_clin = select(df_join,1:ncol(df_clin));
dat_expr = select(df_join, Not(1:ncol(df_clin)));

"""
SAVE DATA
"""

fexpr_out = datadir("processed","TCGA_BRCA","TCGA-BRCA-exp-348-expr.csv");
CSV.write(fexpr_out,dat_expr);

fclin_out = datadir("processed","TCGA_BRCA","TCGA-BRCA-exp-348-clin.csv");
CSV.write(fclin_out,dat_clin);