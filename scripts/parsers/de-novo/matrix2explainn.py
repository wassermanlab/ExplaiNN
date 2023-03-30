#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import pickle
from pybedtools import BedTool
from pybedtools.helpers import cleanup
import re
import subprocess as sp
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from utils import get_file_handle

# Globals
scripts_dir = os.path.dirname(os.path.realpath(__file__))

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "matrix_dir", type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "genome_file", type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "regions_idx", type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-d", "--dummy-dir",
    help="Dummy directory.",
    type=click.Path(resolve_path=True),
    default="/tmp/",
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

    # Get already processed TFs
    tfs = set()
    for tsv_file in os.listdir(args["output_dir"]):
        m = re.search("^(\S+).(train|validation|test).tsv.gz$", tsv_file)
        tfs.add(m.group(1))

    # Get matrix files
    matrix_files = []
    for matrix_file in os.listdir(args["matrix_dir"]):
        m = re.search("^matrix2d.(\S+).ReMap.sparse.npz$", matrix_file)
        if m.group(1) not in tfs:
            matrix_files.append(os.path.join(args["matrix_dir"], matrix_file))

    # Get regions idx
    handle = get_file_handle(args["regions_idx"], mode="rb")
    regions_idx = pickle.load(handle)
    handle.close()
    idx_regions = {v: k for k, v in regions_idx.items()}

    # Get FASTA sequences
    kwargs = {"total": len(matrix_files), "bar_format": bar_format}
    pool = Pool(args["threads"])
    p = partial(_to_ExplaiNN, genome_file=args["genome_file"],
        idx_regions=idx_regions, dummy_dir=args["dummy_dir"],
        output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, matrix_files), **kwargs):
        pass

def _to_ExplaiNN(matrix_file, genome_file, idx_regions,
    dummy_dir="/tmp/", output_dir="./"):

    # Initialize
    prefix = re.search("^matrix2d.(\S+).ReMap.sparse.npz$",
        os.path.split(matrix_file)[1]).group(1)

    # Load matrix 2D as numpy array
    matrix2d = np.load(matrix_file)["arr_0"]

    # Get ones and zeros
    matrix1d = np.nanmax(matrix2d, axis=0)
    ones = np.where(matrix1d == 1.)[0]
    zeros = np.where(matrix1d == 0.)[0]

    # Get BedTool objects (i.e. positive/negative sequences)
    b = BedTool("\n".join(["\t".join(map(str, idx_regions[i])) \
            for i in ones]), from_string=True).sort()
    b.sequence(fi=genome_file)
    positive_file = os.path.join(dummy_dir, "%s_pos.fa" % prefix)
    b.save_seqs(positive_file)
    b = BedTool("\n".join(["\t".join(map(str, idx_regions[i])) \
            for i in zeros]), from_string=True).sort()
    b.sequence(fi=genome_file)
    negative_file = os.path.join(dummy_dir, "%s_neg.fa" % prefix)
    b.save_seqs(negative_file)

    # Subsample negative sequences by %GC
    json_file = os.path.join(dummy_dir, "%s.json" % prefix)
    cmd = "./utils/match-seqs-by-gc.py -f -o %s %s %s" % \
        (json_file, negative_file, positive_file)
    _ = sp.run([cmd], shell=True, cwd=scripts_dir, stdout=sp.DEVNULL,
        stderr=sp.DEVNULL)

    # To ExplaiNN
    cmd = "./json2explainn.py -o %s -p %s --test %s" % \
        (output_dir, prefix, json_file)
    _ = sp.run([cmd], shell=True, cwd=scripts_dir, stdout=sp.DEVNULL,
        stderr=sp.DEVNULL)

    # Delete tmp files
    cleanup()
    os.remove(positive_file)
    os.remove(negative_file)
    os.remove(json_file)

if __name__ == "__main__":
    main()