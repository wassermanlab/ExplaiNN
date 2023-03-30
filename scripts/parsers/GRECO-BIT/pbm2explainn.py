#!/usr/bin/env python

import click
from functools import partial
import gzip
from multiprocessing import Pool
import os
import pandas as pd
import re
from sklearn.preprocessing import quantile_transform
import subprocess as sp
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir,
                                os.pardir))
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from utils import get_data_splits

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "intensities_dir", type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-c", "--cpu-threads",
    help="Number of CPU threads to use.",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "-n", "--no-linker",
    help="Exclude the linker sequence.",
    is_flag=True,
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True
)
@click.option(
    "-q", "--quantile-normalize",
    help="Quantile normalize signal intensities.",
    is_flag=True
)

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Get intensity files
    intensity_files = []
    for intensity_file in os.listdir(args["intensities_dir"]):
        intensity_files.append(os.path.join(args["intensities_dir"],
                               intensity_file))

    # Get ExplaiNN files
    kwargs = {"total": len(intensity_files), "bar_format": bar_format}
    pool = Pool(args["cpu_threads"])
    p = partial(_to_ExplaiNN, no_linker=args["no_linker"],
                output_dir=args["output_dir"],
                quantile_normalize=args["quantile_normalize"])
    for _ in tqdm(pool.imap(p, intensity_files), **kwargs):
        pass

def _to_ExplaiNN(intensity_file, no_linker=False, output_dir="./",
                 quantile_normalize=False):

    # Initialize
    prefix = None
  
    # Get prefix
    m = re.search("^(\S+@\S+@\S+)\.5\w+@\S+\.(\S+)\.(\S+)\.tsv$",
                    os.path.basename(intensity_file))
    prefix = m.group(1) + "@" + m.group(2)

    # Get DataFrame
    df = pd.read_table(intensity_file, header=0)
    if quantile_normalize:
        df.iloc[:, 7] = quantile_transform(df.iloc[:, 7].to_numpy().reshape(-1, 1),
            n_quantiles=10, random_state=0, copy=True)
    if not no_linker:
        df["pbm_sequence"] += df["linker_sequence"]
    df = df[["id_probe", "pbm_sequence", "mean_signal_intensity"]]

    # Get data splits
    train, validation, _ = get_data_splits(df, [80, 20, 0], 1714)

    # Save TSV files
    if train is not None:
        if prefix is None:
            tsv_file = os.path.join(output_dir, "train.tsv.gz")
        else:
            tsv_file = os.path.join(output_dir, f"{prefix}.train.tsv.gz")
        train.to_csv(tsv_file, sep="\t", header=False, index=False,
                     compression="gzip")
    if validation is not None:
        if prefix is None:
            tsv_file = os.path.join(output_dir, "validation.tsv.gz")
        else:
            tsv_file = os.path.join(output_dir, f"{prefix}.validation.tsv.gz")
        validation.to_csv(tsv_file, sep="\t", header=False, index=False,
                          compression="gzip")

    # Create set for PWM scoring
    validation_file = os.path.join(output_dir, f"{prefix}.validation.tsv.gz")
    test_file = os.path.join(output_dir, f"{prefix}.pwm-scoring.tsv.gz")
    df = pd.read_table(validation_file, header=None)
    ones = df.nlargest(int(len(df) * .05), 2)
    ones[2] = 1.
    zeros = df.nsmallest(int(len(df) * .05), 2)
    zeros[2] = 0.
    df = pd.concat((ones, zeros)).reset_index(drop=True)
    df.to_csv(test_file, sep="\t", header=False, index=False,
              compression="gzip")

if __name__ == "__main__":
    main()