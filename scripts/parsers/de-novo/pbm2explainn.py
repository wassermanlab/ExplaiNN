#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import quantile_transform
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
    "signal_intensities_dir", type=click.Path(exists=True, resolve_path=True)
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

    # Get signal intensity files
    tfs_sig_int_files = []
    df = pd.read_csv(args["tsv_file"], sep="\t")
    df = df.groupby("TF").first().reset_index()
    sig_int_files = {}
    for sig_int_file in os.listdir(args["signal_intensities_dir"]):
        tf = sig_int_file.split("_")[0]
        sig_int_files.setdefault(tf, [])
        sig_int_files[tf].append(
            os.path.join(args["signal_intensities_dir"], sig_int_file)
        )
    for _, row in df.iterrows():
        if row["UniPROBE"] is not None:
            tfs_sig_int_files.append([row["TF"],
                [sorted(sig_int_files[tf])]])

    # Get FASTA sequences
    kwargs = {"total": len(tfs_sig_int_files), "bar_format": bar_format}
    pool = Pool(args["threads"])
    p = partial(_to_ExplaiNN, output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, tfs_sig_int_files), **kwargs):
        pass

def _to_ExplaiNN(tfs_sig_int_files, output_dir="./"):

    # Initialize
    data_splits = ["train", "validation"]
    tf, sig_int_files = tfs_sig_int_files
    rng = np.random.RandomState(0)

    # For each data split, signal intensity file...
    for data_split, sign_int_file in zip(data_splits, sig_int_files[0]):

        # Read signal intensities
        df = pd.read_csv(sign_int_file, header=None, sep="\t")

        # Quantile normalize intensity signals
        df.iloc[:, 0] = quantile_transform(df.iloc[:, 0].to_numpy().reshape(-1, 1),
            n_quantiles=10, random_state=0, copy=True)
        df["Index"] = df.index
        df = df.iloc[:, ::-1]

        # Save sequences
        tsv_file = os.path.join(output_dir, f"{tf}.{data_split}.tsv.gz")
        df = df.sample(frac=1)
        df.to_csv(tsv_file, sep="\t", header=False, index=False,
            compression="gzip")

if __name__ == "__main__":
    main()