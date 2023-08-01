```bash
# Create conda environment
conda create -y -n explainn -c pytorch -c conda-forge -c bioconda \
    bedops \
    biasaway \
    biopython \
    click click-option-group \
    cudatoolkit=11.0.3 pytorch=1.11 torchaudio=0.12.1 torchvision=0.13.1 \
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
    python=3.9.12 \
    scikit-learn \
    sra-tools=3.0.0 \
    tqdm
```