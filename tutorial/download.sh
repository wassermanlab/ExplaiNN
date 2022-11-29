#!/usr/bin/env bash

# # AI-TAC dataset
# URL=https://www.dropbox.com/s
# if [ ! -f ./AI-TAC/ImmGenATAC1219.peak_matched.txt ]; then
#     wget -P AI-TAC ${URL}/r8drj2wxc07bt4j/ImmGenATAC1219.peak_matched.txt
# fi
# if [ ! -f ./AI-TAC/mouse_peak_heights.csv ]; then
#     wget -P AI-TAC ${URL}/7mmd4v760eux755/mouse_peak_heights.csv
# fi

# ENCODE ATAC-seq data on A-549 cells
URL=https://www.encodeproject.org/files
if [ ! -f ./ENCODE/ENCFF735UWS.bed.gz ]; then
    wget -P ENCODE ${URL}/ENCFF735UWS/@@download/ENCFF735UWS.bed.gz
fi

# HT-SELEX data for TF CEBPB
if [ ! -f ./HT-SELEX/Cycle1/ERR1002403_1.fastq.gz ]; then
        parallel-fastq-dump --sra-id ERR1002403 \
            --threads 8 --outdir ./HT-SELEX/Cycle1 --split-files --gzip
fi
if [ ! -f ./HT-SELEX/Cycle2/ERR1002405_1.fastq.gz ]; then
    parallel-fastq-dump --sra-id ERR1002405 \
        --threads 8 --outdir ./HT-SELEX/Cycle2 --split-files --gzip
fi
if [ ! -f ./HT-SELEX/Cycle3/ERR1002407_1.fastq.gz ]; then
    parallel-fastq-dump --sra-id ERR1002407 \
        --threads 8 --outdir ./HT-SELEX/Cycle3 --split-files --gzip
fi
if [ ! -f ./HT-SELEX/Cycle4/ERR1002409_1.fastq.gz ]; then
    parallel-fastq-dump --sra-id ERR1002409 \
        --threads 8 --outdir ./HT-SELEX/Cycle4 --split-files --gzip
fi

# PBM data for TF Cebpb
URL=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1291nnn
if [ ! -f ./PBM/GSM1291451_pTH5460_ME_8mer_373.raw.txt.gz ]; then
    GSM=GSM1291451
    wget -P PBM ${URL}/${GSM}/suppl/${GSM}_pTH5460_ME_8mer_373.raw.txt.gz
fi
if [ ! -f ./PBM/GSM1291450_pTH5460_HK_8mer_1598.raw.txt.gz ]; then
    GSM=GSM1291450
    wget -P PBM ${URL}/${GSM}/suppl/${GSM}_pTH5460_HK_8mer_1598.raw.txt.gz
fi

# ReMap ChIP-seq data for TFs CEBPB, CTCF, and FOXA1 on A-549 cells
URL=https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET
if [ ! -f ./ReMap/ENCSR000BUB.CEBPB.A-549.bed.gz ]; then
    wget -P ReMap ${URL}/ENCSR000BUB.CEBPB.A-549.bed.gz
fi
if [ ! -f ./ReMap/ENCSR000DPF.CTCF.A-549.bed.gz ]; then
    wget -P ReMap ${URL}/ENCSR000DPF.CTCF.A-549.bed.gz
fi
if [ ! -f ./ReMap/ENCSR000BRD.FOXA1.A-549.bed.gz ]; then
    wget -P ReMap ${URL}/ENCSR000BRD.FOXA1.A-549.bed.gz
fi

# SMiLE-seq data for TF CEBPB
if [ ! -f ./SMiLE-seq/SRR3405054_1.fastq.gz ]; then
    parallel-fastq-dump --sra-id SRR3405054 \
        --threads 8 --outdir ./SMiLE-seq --split-files --gzip
fi
