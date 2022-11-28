#!/usr/bin/env bash

# Create conda environment
conda create -y -n explainn2 -c pytorch -c conda-forge -c bioconda \
    bedops \
    biasaway \
    biopython \
    click click-option-group \
    cudatoolkit=11.3.1 pytorch=1.11.0 torchvision=0.12.0  \
    fastcluster \
    genomepy \
    h5py \
    joblib=1.1.0 \
    logomaker \
    matplotlib \
    numpy \
    pandas \
    parallel-fastq-dump \
    pybedtools \
    python=3.9.12 \
    scikit-learn \
    sra-tools=3.0.0 \
    tqdm

# Activate conda environment
conda activate explainn

# # Download MEME
# URL=https://anaconda.org/bioconda/meme/5.4.1/download/linux-64
# FILE=meme-5.4.1-py310pl5321hb021246_2.tar.bz2
# wget ${URL}/${FILE}
# conda install ${FILE}
# rm ${FILE}
