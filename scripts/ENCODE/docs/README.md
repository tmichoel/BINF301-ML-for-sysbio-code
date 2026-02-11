
# YY1 in GM12878 & K562 with DNase (Dataset A)

This repository builds a **ready-to-use teaching dataset** for reproducing core ideas from Arvey et al., *Genome Research* 2012: k-mer SVM sequence models, DNase spatial signatures, and combined models.

## Contents
- 1000 YY1 peaks in **GM12878** and **K562** (hg19)
- 100 bp **peak-centered sequences** and **matched negatives** (+200 bp shift)
- **DNase-seq** signal summarized in **5 × 100 bp** bins around peak centers
- A unified `labels.tsv` for direct use in ML pipelines

## Data sources (accessions)
- YY1 ChIP–seq (GM12878): ENCSR000EUM / GEO GSM935482 (narrowPeak)
- YY1 ChIP–seq (K562): ENCSR000BMH / GEO GSM935368 (narrowPeak)
- DNase–seq (GM12878): ENCSR000EMT / GEO GSM736620 (bigWig)
- DNase–seq (K562): ENCSR000EKS / GEO GSM816655 (bigWig)
- Reference genome (hg19): UCSC `hg19.fa.gz`

> Exact download URLs and identifiers are saved in `accessions.json`.

## Quick start
```bash
# 1) Create a fresh conda env (recommended)
conda create -n yy1_ds python=3.10 -y && conda activate yy1_ds

# 2) Install dependencies
pip install pandas numpy pyfaidx pybigwig tqdm

# 3) Build the dataset (downloads peaks, DNase, and hg19; then writes FASTA/TSV)
python scripts/build_dataset.py --outdir processed   --n-peaks 1000 --bin-size 100 --n-bins 5 --assembly hg19

# 4) Inspect outputs
ls -lh processed
```

## Output files
See `MANIFEST.json` for the full list and descriptions.

## Notes
- Coordinates and files are **hg19** to match the Arvey et al. 2012 analysis.
- DNase signal is summarized as the mean value per bin from the bigWig tracks.
- Negatives are constructed by shifting +200 bp from each peak center (clipped to chromosome bounds).

## Citation
If you use this dataset structure or builder, please cite:
- Arvey *et al.* 2012, Genome Research 22:1723–1734.
- ENCODE Project data used here (YY1 ChIP–seq, DNase–seq for GM12878 and K562).

