using DrWatson
@quickactivate "BINF301-code"

using DataFrames
using CSV
using Arrow

"""
Expression data

- URL: http://m3d.mssm.edu/norm/
- Download: E_coli_v4_Build_6.tar.gz
- Extract: avg\_E\_coli\_v4\_Build\_6\_exps466probes4297.tab
- Truncate gene names at first \"_\"
"""

fexpr = datadir("raw","ecoli-grn","E_coli_v4_Build_6", "avg_E_coli_v4_Build_6_exps466probes4297.tab");

dfexpr = permutedims(DataFrame(CSV.File(fexpr)),1);
rename!(x -> lowercase(split(x,"_")[1]), dfexpr);
select!(dfexpr, Not(1));

"""
RegulongDB network of known TF - gene interactions
- URL: https://regulondb.ccg.unam.mx/menu/download/datasets/index.jsp
- Download: Regulatory Network Interactions > TF - gene interactions
- Extract file: network\_tf\_gene.txt
- Remove all comment lines (start with #) and rename file to regulonDB\_network\_tf\_gene.txt
- Extract 2nd (TFs) and 4th (target genes) columns, change TF names to start with lowercase
"""

fnet = datadir("raw","ecoli-grn","regulonDB_network_tf_gene.txt");
dfnet = DataFrame(CSV.File(fnet, header=false))
select!(dfnet, :Column2 => (x -> lowercase.(x)), :Column4 => (x -> lowercase.(x)))
rename!(dfnet,[1 => :TF, 2 => :target])
unique!(dfnet)

# Align datasets along common genes
genes_net = intersect(names(dfexpr), union(dfnet.:TF,dfnet.:target))

df = select(dfexpr, genes_net)

iTF = indexin(dfnet[:,1],genes_net)
itgt = indexin(dfnet[:,2],genes_net)
rws = (.!isnothing.(iTF) .& .!isnothing.(itgt)) .& (iTF != itgt)

refnet = DataFrame(:TF => genes_net[iTF[rws]], :iTF => iTF[rws],  :target => genes_net[itgt[rws]], :itgt => itgt[rws])

#sparse(iTF[rws],itgt[rws],ones(sum(rws)),length(genes_net),length(genes_net))

# Save the processed data
fexpr_out = datadir("processed","Ecoli_GRN","avg_E_coli_v4_Build_6_exps466probes4297_filtered.arrow")
fnet_out = datadir("processed","Ecoli_GRN","regulonDB_network_tf_gene_filtered.arrow")

Arrow.write(fexpr_out, df)
Arrow.write(fnet_out, refnet)