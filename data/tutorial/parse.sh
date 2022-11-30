#!/usr/bin/env bash

PARSERS_DIR=../../scripts/parsers

# HT-SELEX
if [ ! -f ./HT-SELEX/train.tsv.gz ]; then
    ${PARSERS_DIR}/fastq2explainn.py -o ./HT-SELEX \
        --splits 80 20 0 \
        -f1 ./HT-SELEX/Cycle1/ERR1002403_1.fastq.gz \
        -f1 ./HT-SELEX/Cycle2/ERR1002405_1.fastq.gz \
        -f1 ./HT-SELEX/Cycle3/ERR1002407_1.fastq.gz \
        -f1 ./HT-SELEX/Cycle4/ERR1002409_1.fastq.gz
fi

# PBM
if [ ! -f ./PBM/train.tsv.gz ]; then
    ${PARSERS_DIR}/pbm2explainn.py -o ./PBM -q \
        --splits 100 0 0 \
        ./PBM/GSM1291450_pTH5460_HK_8mer_1598.raw.txt.gz
fi
if [ ! -f ./PBM/validation.tsv.gz ]; then
    ${PARSERS_DIR}/pbm2explainn.py -o ./PBM -q \
        --splits 0 100 0 \
        ./PBM/GSM1291451_pTH5460_ME_8mer_373.raw.txt.gz
fi

# ReMap
GENOME=../genomes/hg38/hg38.fa
# 1) negative sequences = dinucleotide shuffling
TF="CEBPB"
if [ ! -f ./ReMap/CEBPB.fa ]; then
    zless ./ReMap/ENCSR000BUB.CEBPB.A-549.bed.gz | \
        awk '{print $1"\t"$7-100"\t"$8+100}' > ./ReMap/${TF}_201bp.bed
    bedtools getfasta -fi ${GENOME} -fo ./ReMap/${TF}.fa \
        -bed ./ReMap/${TF}_201bp.bed
fi
if [ ! -f ./ReMap/${TF}.train.tsv.gz ]; then
    ${PARSERS_DIR}/fasta2explainn.py -o ./ReMap/ -p ${TF} \
        --splits 80 20 0 \
        ./ReMap/${TF}.fa
fi
# 2) negative sequences = subsampled open, unbound regions
if [ ! -f ./ReMap/ATAC-seq_nov.fa ]; then
    bedtools subtract -a ENCODE/ENCFF735UWS.bed.gz \
        -b ReMap/ENCSR000BUB.CEBPB.A-549.bed.gz -A | \
        awk '{SUMMIT=$2+$10;print $1"\t"SUMMIT-100-1"\t"SUMMIT+100}' > ./ReMap/ATAC-seq_nov_201bp.bed
    bedtools getfasta -fi ${GENOME} -fo ./ReMap/ATAC-seq_nov.fa \
        -bed ./ReMap/ATAC-seq_nov_201bp.bed
fi
if [ ! -f ./ReMap/${TF}+ATAC-seq_nov.json.gz ]; then
    ${PARSERS_DIR}/match-seqs-by-gc.py -o ./ReMap/${TF}+ATAC-seq_nov.json \
        ./ReMap/${TF}.fa \
        ./ReMap/ATAC-seq_nov.fa
    gzip ./ReMap/${TF}+ATAC-seq_nov.json
fi
if [ ! -f ./ReMap/${TF}+ATAC-seq_nov.train.tsv.gz ]; then
    ${PARSERS_DIR}/json2explainn.py -o ./ReMap/ -p ${TF}+ATAC-seq_nov \
        --splits 80 20 0 \
        ./ReMap/${TF}+ATAC-seq_nov.json.gz
fi

# TF="CTCF"
# if [ ! -f ./ReMap/${TF}.fa ]; then
#     zless ./ReMap/ENCSR000DPF.CTCF.A-549.bed.gz | \
#         awk '{print $1"\t"$7-100"\t"$8+100}' > ./ReMap/${TF}_201bp.bed
#         bedtools getfasta -fi ${HG38} -fo ./ReMap/${TF}.fa \
#         -bed ./ReMap/${TF}_201bp.bed
#     ../scripts/parsers/subsample-seqs-by-gc.py -s 30000 \
#         -o ./ReMap/${TF}_subsampled.fa ./ReMap/${TF}.fa
# fi
# TF="FOXA1"
# if [ ! -f ./ReMap/FOXA1.fa ]; then
#     zless ./ReMap/ENCSR000BRD.FOXA1.A-549.bed.gz | \
#         awk '{print $1"\t"$7-100"\t"$8+100}' > ./ReMap/${TF}_201bp.bed
#         bedtools getfasta -fi ${HG38} -fo ./ReMap/${TF}.fa \
#         -bed ./ReMap/${TF}_201bp.bed
#     ../scripts/parsers/subsample-seqs-by-gc.py -s 30000 \
#         -o ./ReMap/${TF}_subsampled.fa ./ReMap/${TF}.fa
# fi
# TF="GATA3"
# if [ ! -f ./ReMap/GATA3.fa ]; then
#     zless ./ReMap/ENCSR000BTI.GATA3.A-549.bed.gz | \
#         awk '{print $1"\t"$7-100"\t"$8+100}' > ./ReMap/${TF}_201bp.bed
#         bedtools getfasta -fi ${HG38} -fo ./ReMap/${TF}.fa \
#         -bed ./ReMap/${TF}_201bp.bed
#     ../scripts/parsers/subsample-seqs-by-gc.py -s 30000 \
#         -o ./ReMap/${TF}_subsampled.fa ./ReMap/${TF}.fa
# fi
# TF="JUND"
# if [ ! -f ./ReMap/JUND.fa ]; then
#     zless ./ReMap/ENCSR000BRF.JUND.A-549.bed.gz | \
#         awk '{print $1"\t"$7-100"\t"$8+100}' > ./ReMap/${TF}_201bp.bed
#         bedtools getfasta -fi ${HG38} -fo ./ReMap/${TF}.fa \
#         -bed ./ReMap/${TF}_201bp.bed
#     ../scripts/parsers/subsample-seqs-by-gc.py -s 30000 \
#         -o ./ReMap/${TF}_subsampled.fa ./ReMap/${TF}.fa
# fi
# if [ ! -f ./ReMap/GATA3+DHS.json ]; then
#     ../scripts/parsers/match-seqs-by-gc.py -o ./ReMap/GATA3+DHS.json \
#         ./ReMap/GATA3.fa ./ENCODE/DHS.fa 
# fi
# if [ ! -f ./ReMap/CTCF+FOXA1+GATA3+JUND_matched.json ]; then
#     ../scripts/parsers/match-seqs-by-gc.py \
#         -o ./ReMap/CTCF+FOXA1+GATA3+JUND_matched.json \
#         ./ReMap/CTCF.fa ./ReMap/FOXA1.fa ./ReMap/GATA3.fa ./ReMap/JUND.fa 
# fi
# if [ ! -f ./ReMap/reference.bed ]; then
#     less ./ReMap/*_201bp.bed | \
#         bedmap --echo-map-range --fraction-both 0.5 --delim "\t" - | \
#         sort-bed - | uniq | \
#         awk '{print $1"\t"int($2+(($3-$2)/2))-100"\t"int($2+(($3-$2)/2))+100+1}' \
#         > ./ReMap/reference.bed
# fi
# ../scripts/parsers/fasta2explainn.py -o ./ReMap \
#     -p GATA3.shuffle ./ReMap/GATA3.fa
# i.e. GATA3 JSON
# ../scripts/parsers/fasta2explainn.py -o ./ReMap \
#     -p CTCF+FOXA1+GATA3+JUND_subsampled \
#     ./ReMap/CTCF.fa ./ReMap/FOXA1.fa ./ReMap/GATA3.fa ./ReMap/JUND.fa
# i.e. GATA3 JSON

# SMiLE-seq
if [ ! -f ./SMiLE-seq/train.tsv.gz ]; then
    ${PARSERS_DIR}/fastq2explainn.py -o ./SMiLE-seq \
        --clip-left 7 \
        --clip-right 64 \
        --splits 80 20 0 \
        -f1 ./SMiLE-seq/SRR3405054_1.fastq.gz
fi
