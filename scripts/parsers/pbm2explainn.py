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
                                os.pardir))

from utils import click_validator, get_data_splits

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

def validate_click_options(context):

    # Check that the data splits add to 100
    v = sum(context.params["splits"])
    if v != 100:
        raise click.BadParameter(f"data splits do not add to 100: {v}.")

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS,
               cls=click_validator(validate_click_options))
@click.argument(
    "intensity_file",
    type=click.Path(exists=True, resolve_path=True),
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
    "-p", "--prefix",
    help="Output prefix.",
    type=str
)
@click.option(
    "-q", "--quantile-normalize",
    help="Quantile normalize signal intensities.",
    is_flag=True
)
@click.option(
    "-r", "--random-seed",
    help="Random seed.",
    type=int,
    default=1714,
    show_default=True
)
@click.option(
    "-s", "--splits",
    help="Training, validation and test data splits.",
    nargs=3,
    type=click.IntRange(0, 100),
    default=[80, 10, 10],
    show_default=True
)

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Get TSV files for ExplaiNN
    _to_ExplaiNN(args["intensity_file"], args["no_linker"], args["output_dir"],
        args["prefix"], args["quantile_normalize"], args["random_seed"],
        args["splits"])

def _to_ExplaiNN(intensity_file, no_linker=False, output_dir="./", prefix=None,
                 quantile_normalize=False, random_seed=1714,
                 splits=[80, 10, 10]):

    # Initialize
    data = []
  
    # Get DataFrame
    df = pd.read_table(intensity_file, header=0)
    if quantile_normalize:
        df.iloc[:, 7] = quantile_transform(df.iloc[:, 7].to_numpy().reshape(-1, 1),
            n_quantiles=10, random_state=0, copy=True)
    if not no_linker:
        df["pbm_sequence"] += df["linker_sequence"]
    df = df[["id_probe", "pbm_sequence", "mean_signal_intensity"]].dropna()

    # Get data splits
    train, validation, test = get_data_splits(df, splits, random_seed)

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
    if test is not None:   
        if prefix is None:
            tsv_file = os.path.join(output_dir, "test.tsv.gz")
        else:
            tsv_file = os.path.join(output_dir, f"{prefix}.test.tsv.gz")
        test.to_csv(tsv_file, sep="\t", header=False, index=False,
                    compression="gzip")

if __name__ == "__main__":
    main()