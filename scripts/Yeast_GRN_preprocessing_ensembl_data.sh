#!/bin/sh
#  Script by Adriaan Ludl, original here: https://github.com/michoel-lab/FindrCausalNetworkInferenceOnYeast/blob/main/scripts/5_preprocessing_ensembl_data.sh, updated to Ensemble v 111
#  Processing of Ensembl file listing for yeast:
#
echo '# Start processing yeast ensembl data'

inpath=../data/raw/Yeast_GRN/Ensembl/
outpath=../data/raw/Yeast_GRN/Ensembl/

file1=Saccharomyces_cerevisiae.R64-1-1.111.gff3
out1=cleaned_genes_ensembl111_yeast_step1.txt
out2=cleaned_genes_ensembl111_yeast_step2.txt

sed -n -e '/gene/{p;n;}' $inpath$file1 | sed -e 's/;/ /g' -e 's/description.*/ /' > $outpath$out1
#sed  -e 's/:/ /g' -e "s/[[:space:]]\+/ /g" $outpath$out1 > $outpath$out2
sed -n  '/ID=gene/p' $outpath$out1 | sed -n  '/Name/p'   > $outpath$out2

echo '# Done'
# EOF
