#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import os
import pandas as pd
import re
import subprocess as sp
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir,
                                os.pardir))
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from parsers.fastq2explainn import _to_ExplaiNN

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "reads_dir", type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-c", "--cpu-threads",
    help="Number of CPU threads to use.",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True
)

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Get reads files
    reads_files = []
    for reads_file in os.listdir(args["reads_dir"]):
        reads_files.append(os.path.join(args["reads_dir"], reads_file))

    # Group reads files from different cycles
    grouped_reads_files = {}
    for reads_file in sorted(reads_files):
        m = re.search("^(\S+@\S+@\S+)\.C\d\.5\w+\.3\w+@\S+",
                      os.path.basename(reads_file))
        grouped_reads_files.setdefault(m.group(1), list())
        grouped_reads_files[m.group(1)].append(reads_file)
    grouped_reads_files = list(grouped_reads_files.values())

    # Get ExplaiNN files
    kwargs = {"total": len(grouped_reads_files), "bar_format": bar_format}
    pool = Pool(args["cpu_threads"])
    p = partial(_get_ExplaiNN_files, output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, grouped_reads_files), **kwargs):
        pass

def _get_ExplaiNN_files(reads_files, output_dir="./"):
    """
    Naming format:
    AKAP8L@silly-willy+clever-peter@Megaman.HOMER@motif-id123.pcm
    """

    # Initialize
    prefix = None
    suffixes = []

    # For each read file...
    for i, reads_file in enumerate(sorted(reads_files)):

        # Get prefix and sufix
        m = re.search("^(\S+@\S+@\S+)\.C\d\.5\w+\.3\w+@\S+\.(\S+)\.\S+\.fastq",
                      os.path.basename(reads_file))
        if prefix is None:
            prefix = m.group(1)
        suffixes.append(m.group(2))

    # Get prefix
    prefix += "@%s" % "+".join(suffixes)

    # Create train and validation sets
    _to_ExplaiNN(clip_left=None, clip_right=None, dummy_dir="/tmp/",
                 fastq_1=reads_files, fastq_2=[], output_dir=output_dir,
                 prefix=prefix, random_seed=1714, splits=[80, 20, 0])

    # Create set for PWM scoring
    validation_file = os.path.join(output_dir, f"{prefix}.validation.tsv.gz")
    test_file = os.path.join(output_dir, f"{prefix}.pwm-scoring.tsv.gz")
    df = pd.read_table(validation_file, header=None)
    zeros = [1.] + [0. for i in range(len(df.columns) - 3)]
    ones = [0. for i in range(len(df.columns) - 3)] + [1.]
    sub_df = df.iloc[:, 2:]
    zeros = sub_df[sub_df == zeros].dropna()
    ones = sub_df[sub_df == ones].dropna()
    zeros = df.loc[zeros.index].iloc[:, :2]
    zeros[2] = 0.
    ones = df.loc[ones.index].iloc[:, :2]
    ones[2] = 1.
    df = pd.concat((zeros, ones)).reset_index(drop=True)
    df.to_csv(test_file, sep="\t", header=False, index=False,
              compression="gzip")

if __name__ == "__main__":
    main()