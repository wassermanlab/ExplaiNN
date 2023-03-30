#!/usr/bin/env python

from Bio import SeqIO
from Bio.Seq import Seq
try:
    from Bio.SeqUtils import gc_fraction
except:
    from Bio.SeqUtils import GC as gc_fraction
import click
import copy
import importlib
import json
import random
import re
import sys

lib = importlib.import_module("subsample-seqs-by-gc")

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "fasta_file",
    type=click.Path(exists=True, resolve_path=True),
    nargs=-1,
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
    help="Output file (in JSON format).  [default: STDOUT]",
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
)
@click.option(
    "-t", "--transform",
    type=click.Choice(["skip", "shuffle", "mask"]),
    help="Skip, shuffle, or mask (i.e. convert to Ns) DNA.",
    show_default=True
)

def cli(**args):

    # Group sequences based on their %GC content
    gc_groups = _get_GC_groups(args["fasta_file"], args["dna"],
                               args["transform"])

    # Match sequences based on their %GC content
    matched_seqs = _match_seqs_by_GC(gc_groups, args["random_seed"])

    # Subsample sequences based on their %GC content
    if args["subsample"]:
        sampled_seqs = lib._subsample_seqs_by_GC(matched_seqs,
                                                 args["random_seed"],
                                                 abs(args["subsample"]))
    else:
        sampled_seqs = matched_seqs
    sampled_seqs.insert(0, ["labels"] + list(args["fasta_file"]))

    # Write
    if args["output_file"] is not None:
        handle = open(args["output_file"], "wt")
    else:
        handle = sys.stdout
    json.dump(sampled_seqs, handle, indent=4, sort_keys=True)
    handle.close()

def _get_GC_groups(fasta_files, dna="lowercase", transform=None):

    # Initialize
    gc_groups = {}
    if dna == "lowercase":
        regexp = re.compile(r"[^ACGT]+")
    else:
        regexp = re.compile(r"[^acgt]+")

    # For each FASTA file
    for i in range(len(fasta_files)):

        fasta_file = fasta_files[i]

        # For each SeqRecord...
        for record in SeqIO.parse(fasta_file, "fasta"):

            gc = round(gc_fraction(record.seq))

            if transform:

                s = str(record.seq)

                # Skip
                if transform == "skip":
                    if re.search(regexp, s):
                        continue

                # Shuffle/Mask
                else:
                    # 1) extract blocks of nucleotides matching regexp;
                    # 2) either shuffle them or create string of Ns;
                    # and 3) put the nucleotides back
                    l = list(s)
                    for m in re.finditer(regexp, s):
                        if transform == "shuffle":
                            sublist = l[m.start():m.end()]
                            random.shuffle(sublist)
                            l[m.start():m.end()] = copy.copy(sublist)
                        else:
                            l[m.start():m.end()] = "N" * (m.end() - m.start())
                    record.seq = Seq("".join(l))

            # Group SeqRecords based on their %GC content
            gc_groups.setdefault(gc, [[] for i in range(len(fasta_files))])
            gc_groups[gc][i].append(record)

    return(gc_groups)

def _match_seqs_by_GC(gc_groups, random_seed=1714):

    # Initialize
    matched_seqs = []

    # For each %GC content group...
    for i in sorted(gc_groups):

        # For each set of sequences...
        for j in range(len(gc_groups[i])):

            # Shuffle sequences
            random.Random(random_seed).shuffle(gc_groups[i][j])

        # Get the smallest number of sequences in %GC content group
        min_len = min(len(gc_groups[i][j]) for j in range(len(gc_groups[i])))

        # Sequence counter
        for k in range(min_len):

            matched_seqs.append([i])

            # For each set of sequences...
            for j in range(len(gc_groups[i])):

                record = gc_groups[i][j][k]
                matched_seqs[-1].extend([[record.id, str(record.seq)]])

    return(matched_seqs)

if __name__ == "__main__":
    cli()