#!/usr/bin/env bash

if [ ${HOSTNAME} == "gpurtx-2" ]; then
    PYTORCH="pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1"
    CUDATOOLKIT="cudatoolkit=11.3"
    PYTHON="python=3.9"
elif [ ${HOSTNAME} == "GPURTX" ]; then
    PYTORCH="pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2"
    CUDATOOLKIT="cudatoolkit=11.0"
    PYTHON="python=3.9"
else
    continue
fi

# Create conda environment
conda create -y -n explainn -c pytorch -c conda-forge -c bioconda \
    bedops \
    biasaway \
    biopython \
    click click-option-group \
    ${PYTORCH} ${CUDATOOLKIT} \
    fastcluster \
    genomepy \
    h5py \
    joblib=1.1.0 \
    jupyterlab \
    logomaker \
    matplotlib \
    numpy \
    pandas \
    parallel-fastq-dump \
    pybedtools \
    ${PYTHON} \
    scikit-learn \
    sra-tools=3.0.0 \
    tqdm
