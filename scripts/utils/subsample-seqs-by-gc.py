#!/usr/bin/env python

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import click
import importlib
import math
import random
import sys

lib = importlib.import_module("match-seqs-by-gc")

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "fasta_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-d", "--dna",
    type=click.Choice(["uppercase", "lowercase"]),
    help="DNA to transform.",
    show_default=True,
    default="lowercase"
)
@click.option(
    "-o", "--output-file",
    help="Output file (in FASTA format).  [default: STDOUT]",
    type=click.Path(writable=True, readable=False, resolve_path=True,
        allow_dash=True),
)
@click.option(
    "-r", "--random-seed",
    help="Random seed.",
    type=int,
    default=1714,
    show_default=True
)
@click.option(
    "-s", "--subsample",
    help="Number of sequences to subsample.",
    type=int,
    default=1000,
    show_default=True,
)
@click.option(
    "-t", "--transform",
    type=click.Choice(["skip", "shuffle", "mask"]),
    help="Skip, shuffle, or mask (i.e. convert to Ns) DNA.",
    show_default=True
)

def main(**args):

    # Group sequences based on their %GC content
    gc_groups = lib._get_GC_groups([args["fasta_file"]], args["dna"],
                                   args["transform"])

    # Match sequences based on their %GC content
    matched_seqs = lib._match_seqs_by_GC(gc_groups, args["random_seed"])

    # Subsample sequences based on their %GC content
    if args["subsample"]:
        sampled_seqs = _subsample_seqs_by_GC(matched_seqs, args["random_seed"],
                                             abs(args["subsample"]))
    else:
        sampled_seqs = matched_seqs
    sampled_seqs = [SeqRecord(Seq(arr[1][1]), id=arr[1][0], description="") \
        for arr in sampled_seqs]

    # Write
    if args["output_file"] is not None:
        handle = open(args["output_file"], "wt")
    else:
        handle = sys.stdout
    SeqIO.write(sampled_seqs, handle, "fasta")
    handle.close()

def _subsample_seqs_by_GC(matched_seqs, random_seed=1714, subsample=1000):

    # Initialize
    gc_regroups = {}
    sampled_seqs = []

    # Regroup sequences based on their %GC content
    for arr in matched_seqs:
        gc_regroups.setdefault(arr[0], [])
        gc_regroups[arr[0]].append(arr)

    # Get normalization factor
    norm_factor = subsample / sum([len(v) for v in gc_regroups.values()])

    # Subsample sequences based on their %GC content
    for i in sorted(gc_regroups):
        random.Random(random_seed).shuffle(gc_regroups[i])
        arr = gc_regroups[i][:math.ceil(len(gc_regroups[i])*norm_factor)]
        sampled_seqs.extend(arr)

    # Randomize
    random.Random(random_seed).shuffle(sampled_seqs)

    return(sampled_seqs[:subsample])

if __name__ == "__main__":
    main()
