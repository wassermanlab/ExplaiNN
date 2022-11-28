#!/usr/bin/env python

from Bio import SeqIO
import click
import copy
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.preprocessing import OneHotEncoder
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import subprocess as sp

# Locals
from utils import click_validator, get_file_handle, get_data_splits

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
    "fasta_file",
    type=click.Path(exists=True, resolve_path=True),
    nargs=-1,
)
@click.option(
    "-d", "--dummy-dir",
    help="Dummy directory.",
    type=click.Path(resolve_path=True),
    default="/tmp/",
    show_default=True
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
    _to_ExplaiNN(args["fasta_file"], args["dummy_dir"], args["non_standard"],
        args["output_dir"], args["prefix"], args["random_seed"], args["splits"])

def _to_ExplaiNN(fasta_files, dummy_dir="/tmp/", non_standard=None,
                 output_dir="./", prefix=None, random_seed=1714,
                 splits=[80, 10, 10]):

    # Initialize
    data = []
    regexp = re.compile(r"[^ACGT]+")

    # Ys
    enc = OneHotEncoder()
    arr = np.array(list(range(len(fasta_files)))).reshape(-1, 1)
    enc.fit(arr)
    ys = enc.transform(arr).toarray().tolist()

    # Get DataFrame
    for i, fasta_file in enumerate(fasta_files):
        handle = get_file_handle(fasta_file, "rt")
        for record in SeqIO.parse(handle, "fasta"):
            s = str(record.seq)
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
        cwd = os.path.dirname(os.path.realpath(__file__))
        dummy_file = os.path.join(dummy_dir, "%s+%s+%s.fa" %
            (os.path.split(__file__)[1], str(os.getpid()), prefix))
        with open(dummy_file, "wt") as handle:
            for z in zip(df.iloc[:, 0]. tolist(), df.iloc[:, 1]. tolist()):
                handle.write(f">{z[0]}\n{z[1]}\n")
        cmd = "biasaway k -f %s -k 2 -e 1 > %s.biasaway" % (dummy_file,
            dummy_file)
        _ = sp.run([cmd], shell=True, cwd=cwd, stderr=sp.DEVNULL)
        for s in SeqIO.parse("%s.biasaway" % dummy_file, "fasta"):
            header = "%s::shuf" % s.description.split(" ")[-1]
            data.append([header, str(s.seq), 0.])
        df2 = pd.DataFrame(data, columns=list(range(len(data[0]))))
        df = pd.concat((df, df2))
        os.remove(dummy_file)
        os.remove("%s.biasaway" % dummy_file)

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
