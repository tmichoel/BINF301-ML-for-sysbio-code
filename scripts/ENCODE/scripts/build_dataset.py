#!/usr/bin/env python3
import os, gzip, argparse, json, shutil, tarfile
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
from tqdm import tqdm
import pybigwig

# Lightweight FASTA reader backed by dict of sequences from hg19.fa.gz
class Fasta:
    def __init__(self, fa_gz_path):
        self.seqs = {}
        import gzip
        with gzip.open(fa_gz_path, 'rt') as fh:
            chrom = None
            buf = []
            for line in fh:
                if line.startswith('>'):
                    if chrom:
                        self.seqs[chrom] = ''.join(buf)
                    chrom = line[1:].strip().split()[0]
                    buf = []
                else:
                    buf.append(line.strip().upper())
            if chrom:
                self.seqs[chrom] = ''.join(buf)
    def fetch(self, chrom, start, end):
        s = max(0, start)
        e = min(len(self.seqs[chrom]), end)
        sub = self.seqs[chrom][s:e]
        # pad if clipped
        if e - s < (end - start):
            sub = sub + 'N' * ((end - start) - (e - s))
        return sub

def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    urlretrieve(url, dest)
    return dest


def choose_top_n(narrowpeak_gz, n=1000):
    cols = ['chrom','start','end','name','score','strand','signalValue','pValue','qValue','peak']
    df = pd.read_csv(narrowpeak_gz, sep='\t', header=None, names=cols, compression='gzip')
    df = df.sort_values('signalValue', ascending=False).head(n).reset_index(drop=True)
    return df


def summit_center(row):
    if row['peak'] >= 0:
        return int(row['start'] + row['peak'])
    return int((row['start'] + row['end'])//2)


def write_bed(df, path):
    with open(path,'w') as f:
        for _,r in df.iterrows():
            f.write(f"{r['chrom']}\t{int(r['start'])}\t{int(r['end'])}\t{r['name']}\t{int(r['score'])}\t{r['strand']}\n")


def write_fasta(df, fa, win, path):
    half = win//2
    with open(path,'w') as out:
        for i,r in df.iterrows():
            center = summit_center(r)
            seq = fa.fetch(r['chrom'], center-half, center+half)
            out.write(f">{r['chrom']}:{center-half}-{center+half}\n{seq}\n")


def write_negatives(df, fa, win, shift, path):
    half = win//2
    with open(path,'w') as out:
        for i,r in df.iterrows():
            center = summit_center(r) + shift
            seq = fa.fetch(r['chrom'], center-half, center+half)
            out.write(f">{r['chrom']}:{center-half}-{center+half}\n{seq}\n")


def dnase_bins(df, bw_path, win, n_bins, out_path):
    half = win//2
    step = win//n_bins
    bw = pybigwig.open(bw_path)
    rows = []
    for i,r in tqdm(df.iterrows(), total=len(df)):
        center = summit_center(r)
        vals = []
        for b in range(n_bins):
            s = center - half + b*step
            e = s + step
            v = np.nanmean(bw.values(r['chrom'], s, e))
            if np.isnan(v): v = 0.0
            vals.append(v)
        rows.append([r['chrom'], center - half, center + half] + vals)
    cols = ['chrom','start','end'] + [f'bin_{i}' for i in range(n_bins)]
    pd.DataFrame(rows, columns=cols).to_csv(out_path, sep='\t', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='processed')
    ap.add_argument('--n-peaks', type=int, default=1000)
    ap.add_argument('--bin-size', type=int, default=100)
    ap.add_argument('--n-bins', type=int, default=5)
    ap.add_argument('--assembly', default='hg19')
    args = ap.parse_args()

    with open('accessions.json') as fh:
        acc = json.load(fh)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Download inputs
    gm_np = download(acc['yy1_gm12878_chip']['narrowpeak_url'], 'raw/yy1_gm12878.narrowPeak.gz')
    k562_np = download(acc['yy1_k562_chip']['narrowpeak_url'], 'raw/yy1_k562.narrowPeak.gz')
    gm_bw = download(acc['dnase_gm12878']['bigwig_url'], 'raw/dnase_gm12878.bw')
    k562_bw = download(acc['dnase_k562']['bigwig_url'], 'raw/dnase_k562.bw')
    fa = download(acc['reference_genome']['fasta_url'], 'raw/hg19.fa.gz')

    fasta = Fasta(str(fa))

    # Process peaks
    gm_df = choose_top_n(gm_np, args.n_peaks)
    k_df = choose_top_n(k562_np, args.n_peaks)

    write_bed(gm_df, outdir/'peaks_gm12878_top1000.bed')
    write_bed(k_df, outdir/'peaks_k562_top1000.bed')

    # Sequences and negatives
    win = args.bin_size * args.n_bins
    write_fasta(gm_df, fasta, win, outdir/'sequences_gm12878_peaks_100bp.fa')
    write_fasta(k_df, fasta, win, outdir/'sequences_k562_peaks_100bp.fa')
    write_negatives(gm_df, fasta, win, shift=200, path=outdir/'negatives_gm12878_100bp.fa')
    write_negatives(k_df, fasta, win, shift=200, path=outdir/'negatives_k562_100bp.fa')

    # DNase bins
    dnase_bins(gm_df, str(gm_bw), win, args.n_bins, outdir/'dnase_gm12878_5x100bp.tsv')
    dnase_bins(k_df, str(k562_bw), win, args.n_bins, outdir/'dnase_k562_5x100bp.tsv')

    # Labels file (simple: peaks vs flanks)
    labels = []
    for chrom,start,end,name,score,strand,signal,p,pv,qv,pk in gm_df[['chrom','start','end','name','score','strand','signalValue','pValue','qValue','peak']].itertuples(index=False):
        center = int(start+pk) if pk>=0 else int((start+end)//2)
        labels.append(['GM12878','YY1',f'{chrom}:{center}',1])
        labels.append(['GM12878','YY1',f'{chrom}:{center+200}',0])
    for chrom,start,end,name,score,strand,signal,p,pv,qv,pk in k_df[['chrom','start','end','name','score','strand','signalValue','pValue','qValue','peak']].itertuples(index=False):
        center = int(start+pk) if pk>=0 else int((start+end)//2)
        labels.append(['K562','YY1',f'{chrom}:{center}',1])
        labels.append(['K562','YY1',f'{chrom}:{center+200}',0])
    pd.DataFrame(labels, columns=['cell','tf','locus','label']).to_csv(outdir/'labels.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
