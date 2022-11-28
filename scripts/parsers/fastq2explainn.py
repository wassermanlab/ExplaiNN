#!/usr/bin/env python

from Bio import SeqIO
import click
import copy
from itertools import zip_longest
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.preprocessing import OneHotEncoder
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))

from utils import (click_validator, get_file_handle, get_data_splits,
                   shuffle_string)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

def validate_click_options(context):

    # Check that the data splits add to 100
    v = sum(context.params["splits"])
    if v != 100:
        raise click.BadParameter(f"data splits do not add to 100: {v}.")

    # Check that the FASTQ files are paired
    f1 = len(context.params["fastq_1"])
    f2 = len(context.params["fastq_2"])
    if f2 > 0 and f1 != f2:
        raise click.BadParameter("the FASTQ files are not paired: " + \
                                f"{f1} Read 1 files vs. " + \
                                f"{f2} Read 2 files.")

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS,
               cls=click_validator(validate_click_options))
@click.option(
    "--clip-left",
    help="Trim the leftmost n bases.",
    type=int,
    default=None,
    show_default=True
)
@click.option(
    "--clip-right",
    help="Trim the rightmost n bases.",
    type=int,
    default=None,
    show_default=True
)
@click.option(
    "-f1", "--fastq-1",
    help="Read 1 FASTQ file.",
    type=click.Path(exists=True, resolve_path=True),
    multiple=True,
    required=True
)
@click.option(
    "-f2", "--fastq-2",
    help="Read 2 FASTQ file",
    type=click.Path(exists=True, resolve_path=True),
    multiple=True
)
@click.option(
    "-n", "--non-standard",
    type=click.Choice(["skip", "shuffle", "mask"]),
    help="Skip, shuffle, or mask (i.e. convert to Ns) non-standard (i.e. non A, C, G, T) DNA, including lowercase nucleotides.",
    show_default=True
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
    _to_ExplaiNN(args["clip_left"], args["clip_right"], args["fastq_1"],
        args["fastq_2"], args["non_standard"], args["output_dir"],
        args["prefix"], args["random_seed"], args["splits"])

def _to_ExplaiNN(clip_left=None, clip_right=None, fastq_1=[], fastq_2=[],
                 non_standard=None, output_dir="./", prefix=None,
                 random_seed=1714, splits=[80, 10, 10]):

    # Initialize
    data = []
    if clip_right is not None:
        clip_right = -clip_right
    regexp = re.compile(r"[^ACGT]+")

    # Ys
    enc = OneHotEncoder()
    arr = np.array(list(range(len(fastq_1)))).reshape(-1, 1)
    enc.fit(arr)
    ys = enc.transform(arr).toarray().tolist()

    # Get DataFrame
    for i, fastq_files in enumerate(zip_longest(fastq_1, fastq_2)):
        for fastq_file in fastq_files:
            if fastq_file is None:
                continue
            handle = get_file_handle(fastq_file, "rt")
            for record in SeqIO.parse(handle, "fastq"):
                s = str(record.seq)[clip_left:clip_right]
                y = ys[i]
                # Skip non-standard/lowercase
                if non_standard == "skip":
                    if re.search(regexp, s):
                        continue
                # Shuffle/mask non-standard/lowercase
                elif non_standard is not None:
                    # 1) extract blocks of non-standard/lowercase nucleotides;
                    # 2) either shuffle the nucleotides or create string of Ns; and
                    # 3) put the nucleotides back
                    l = list(s)
                    for m in re.finditer(regexp, s):
                        if non_standard == "shuffle":
                            sublist = l[m.start():m.end()]
                            random.shuffle(sublist)
                            l[m.start():m.end()] = copy.copy(sublist)
                        else:
                            l[m.start():m.end()] = "N" * (m.end() - m.start())
                    s = "".join(l)
                data.append([record.id, s] + y)
            handle.close()
    df = pd.DataFrame(data, columns=list(range(len(data[0]))))
    df = df.groupby(1).max().reset_index()
    df = df.reindex(sorted(df.columns), axis=1)

    # Generate negative sequences by dinucleotide shuffling
    if df.shape[1] == 3: # i.e. only one class
        data = []
        for z in zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()):
            s = shuffle_string(z[1], random_seed=random_seed)
            data.append([f"{z[0]}_shuff", s, 0.])
        df2 = pd.DataFrame(data, columns=list(range(len(data[0]))))
        df = pd.concat((df, df2))

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