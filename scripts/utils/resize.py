#!/usr/bin/env python

import click
import pandas as pd
import sys

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "bed_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "chrom_sizes",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "size",
    type=int,
)
@click.option(
    "-o", "--output-file",
    help="Output file (in BED format).  [default: STDOUT]",
    type=click.Path(writable=True, readable=False, resolve_path=True,
        allow_dash=True),
)

def cli(**args):

    # Get BED file as DataFrame
    bed = pd.read_table(args["bed_file"], names=["chrom", "start", "end"],
                        header=None)

    # Get chrom sizes as dict
    sizes = dict.fromkeys(bed["chrom"].to_list(), -1)
    df = pd.read_table(args["chrom_sizes"], names=["chrom", "size"],
                       header=None)
    sizes.update(dict(zip(df["chrom"].to_list(), df["size"].to_list())))

    # Resize intervals
    s = args["size"] / 2.
    bed["center"] = list(map(int, bed["start"] + (bed["end"] - bed["start"]) / 2.))
    bed["start"] = list(map(int, bed["center"] - s))
    bed["end"]  = list(map(int, bed["center"] + s))

    # Filter intervals
    bed = bed[(bed["start"] >= 0) & \
              (bed["end"] <= bed["chrom"].map(lambda x: sizes[x]))]

    # Write
    if args["output_file"] is not None:
        handle = open(args["output_file"], "wt")
    else:
        handle = sys.stdout
    bed.to_csv(handle, columns=["chrom", "start", "end"], header=False,
               index=False, sep="\t")
    handle.close()

if __name__ == "__main__":
    cli()