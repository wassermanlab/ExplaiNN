#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import os
import pandas as pd
import subprocess as sp
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

# Globals
scripts_dir = os.path.dirname(os.path.realpath(__file__))

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

from utils import get_file_handle

# Globals
scripts_dir = os.path.dirname(os.path.realpath(__file__))

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "reads_dir", type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "tsv_file", type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True
)
@click.option(
    "-t", "--threads",
    help="Threads to use.",
    type=int,
    default=1,
    show_default=True
)

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    fastq_dir = os.path.join(args["output_dir"], "FASTQ")
    if not os.path.exists(fastq_dir):
        os.makedirs(fastq_dir)

    # Get reads files
    tfs_reads_files = []
    df = pd.read_csv(args["tsv_file"], sep="\t")
    df = df.groupby("TF").first().reset_index()
    for _, row in df.iterrows():
        if row["SRA"] is not None:
            tfs_reads_files.append([row["TF"], []])
            for sra in row["SRA"].split(";"):
                reads_file = os.path.join(args["reads_dir"], f"{sra}.fastq.gz")
                fastq_file = os.path.join(fastq_dir, f"{sra}.fastq.gz")
                if not os.path.exists(fastq_file):
                    cmd = f"fastp -i {reads_file} -o {fastq_file} -A -G -w 8"
                    _ = sp.run([cmd], shell=True, cwd=scripts_dir,
                        stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                tfs_reads_files[-1][-1].append(fastq_file)

    # Get FASTA sequences
    kwargs = {"total": len(tfs_reads_files), "bar_format": bar_format}
    pool = Pool(args["threads"])
    p = partial(_to_ExplaiNN, output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, tfs_reads_files), **kwargs):
        pass

def _to_ExplaiNN(tf_reads_files, output_dir="./"):

    # Initialize
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    tf, reads_files = tf_reads_files

    # To ExplaiNN
    cmd = "%s/fastq2explainn.py -o %s -p %s %s" % \
        (base_dir, output_dir, tf, " ".join(reads_files))
    _ = sp.run([cmd], shell=True, cwd=scripts_dir, stdout=sp.DEVNULL,
        stderr=sp.DEVNULL)

if __name__ == "__main__":
    main()