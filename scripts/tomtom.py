#!/usr/bin/env python

import click
from click_option_group import optgroup
from functools import partial
import json
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import subprocess as sp
import time
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"
import warnings
warnings.filterwarnings("ignore")

from utils import get_file_handle

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "query_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "target_file",
    type=click.Path(exists=True, resolve_path=True),
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
    show_default=True,
)
@click.option(
    "-t", "--time",
    help="Return the program's running execution time in seconds.",
    is_flag=True,
)
@optgroup.group("Tomtom")
@optgroup.option(
    "--dist",
    help="Distance metric for scoring alignments.",
    type=str,
    default="pearson",
    show_default=True,
)
@optgroup.option(
    "--evalue",
    help="Use E-value threshold.",
    is_flag=True,
)
@optgroup.option(
    "--min-overlap",
    help="Minimum overlap between query and target.",
    type=int,
    default=5,
    show_default=True,
)
@optgroup.option(
    "--motif-pseudo",
    help="Apply pseudocounts to the query and target motifs.",
    type=float,
    default=0.1,
    show_default=True,
)
@optgroup.option(
    "--thresh",
    help="Significance threshold (i.e., do not show worse motifs).",
    type=float,
    default=0.05,
    show_default=True,
)

def main(**args):

    # Start execution
    start_time = time.time()

    # Initialize
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Save exec. parameters as JSON
    json_file = os.path.join(args["output_dir"],
                             f"parameters-{os.path.basename(__file__)}.json")
    handle = get_file_handle(json_file, "wt")
    handle.write(json.dumps(args, indent=4, sort_keys=True))
    handle.close()

    # Create output dirs
    motifs_dir = os.path.join(args["output_dir"], "motifs")
    if not os.path.isdir(motifs_dir):
        os.makedirs(motifs_dir)
    tomtom_dir = os.path.join(args["output_dir"], "tomtom")
    if not os.path.isdir(tomtom_dir):
        os.makedirs(tomtom_dir)   

    # Get motifs
    motifs = []
    _get_motifs(args["query_file"], motifs_dir, args["cpu_threads"])
    for meme_file in os.listdir(motifs_dir):
        motifs.append(os.path.join(motifs_dir, meme_file))

    # Compute Tomtom similarities
    kwargs = {"bar_format": bar_format, "total": len(motifs)}
    pool = Pool(args["cpu_threads"])
    p = partial(_compute_Tomtom_similarities, target_file=args["target_file"],
                tomtom_dir=tomtom_dir, dist=args["dist"], evalue=args["evalue"],
                minover=args["min_overlap"], mpseudo=args["motif_pseudo"],
                thresh=args["thresh"])
    for _ in tqdm(pool.imap(p, motifs), **kwargs):
        pass

    # Save Tomtom file
    tsv_file = os.path.join(args["output_dir"], "tomtom.tsv.gz")
    if not os.path.exists(tsv_file):
        df = _load_Tomtom_files(tomtom_dir)
        df.to_csv(tsv_file, sep="\t", index=False)

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Execution time {seconds} seconds")

def _get_motifs(meme_file, motifs_dir, cpu_threads=1):

    # Initialize
    motifs = []
    parse = False

    # Get motifs
    handle = get_file_handle(meme_file, "rt")
    for line in handle:
        line = line.strip("\n")
        if line.startswith("MOTIF"):
            motifs.append([])
            parse = True
        if parse:
            motifs[-1].append(line)
    handle.close()

    # Create motif files
    zfill = len(str(len(motifs)))
    kwargs = {"bar_format": bar_format, "total": len(motifs)}
    pool = Pool(cpu_threads)
    p = partial(__write_motif, motifs_dir=motifs_dir, zfill=zfill)
    for _ in tqdm(pool.imap(p, enumerate(motifs)), **kwargs):
        pass

def __write_motif(i_motif, motifs_dir, zfill=0):

    # Initialize
    i, motif = i_motif
    prefix = str(i).zfill(zfill)

    motif_file = os.path.join(motifs_dir, f"{prefix}.meme")
    if not os.path.exists(motif_file):
        handle = get_file_handle(motif_file, "wt")
        handle.write("MEME version 4\n\n")
        handle.write("ALPHABET= ACGT\n\n")
        handle.write("strands: + -\n\n")
        handle.write(
            "Background letter frequencies (from uniform background):\n"
        )
        handle.write("A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")
        for line in motif:
            handle.write(f"{line}\n")
        handle.close()

def _compute_Tomtom_similarities(query_file, target_file, tomtom_dir,
                                 dist="pearson", evalue=False, minover=5,
                                 mpseudo=0.1, thresh=0.05):

    # Initialize
    prefix = os.path.splitext(os.path.basename(query_file))[0]
    tomtom_file = os.path.join(tomtom_dir, f"{prefix}.tsv.gz")

    if not os.path.exists(tomtom_file):

        # Compute motif similarities
        if evalue:
            cmd = ["tomtom", "-dist", dist, "-motif-pseudo", str(mpseudo),
                   "-min-overlap", str(minover), "-thresh", str(thresh),
                   "-text", "-evalue", query_file, target_file]
        else:
            cmd = ["tomtom", "-dist", dist, "-motif-pseudo", str(mpseudo),
                   "-min-overlap", str(minover), "-thresh", str(thresh),
                   "-text", query_file, target_file]
        proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)

        # Save Tomtom results
        handle = get_file_handle(tomtom_file, "wb")
        for line in proc.stdout.decode().split("\n"):
            handle.write(f"{line}\n".encode())
        handle.close()

def _load_Tomtom_files(tomtom_dir, col_names=None):

    # Initialize
    dfs = []

    for tsv_file in os.listdir(tomtom_dir):
        if tsv_file.endswith(".tsv.gz"):
            dfs.append(pd.read_table(os.path.join(tomtom_dir, tsv_file),
                                     header=0, usecols=col_names, comment="#"))

    return(pd.concat(dfs))

if __name__ == "__main__":
    main()