#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import os
import pandas as pd
from pybedtools import BedTool
import re
import subprocess as sp
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir,
                                os.pardir))
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from parsers.bed2explainn import _to_ExplaiNN
from utils import shuffle_string

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "intervals_dir", type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "genome_file", type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-c", "--cpu-threads",
    help="Number of CPU threads to use.",
    type=int,
    default=1,
    show_default=True,
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

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Get intervals files
    intervals_files = []
    for intervals_file in os.listdir(args["intervals_dir"]):
        intervals_files.append(os.path.join(args["intervals_dir"],
                               intervals_file))

    # Get ExplaiNN files
    kwargs = {"total": len(intervals_files), "bar_format": bar_format}
    pool = Pool(args["cpu_threads"])
    p = partial(_get_ExplaiNN_file, genome_file=args["genome_file"],
                dummy_dir=args["dummy_dir"], output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, intervals_files), **kwargs):
        pass

def _get_ExplaiNN_file(intervals_file, genome_file, dummy_dir="/tmp/",
                       output_dir="./"):
    """
    Naming format:
    AKAP8L@silly-willy+clever-peter@Megaman.HOMER@motif-id123.pcm
    """

    # Initialize
    prefix = None

    # Get prefix
    m = re.search("^(\S+@\S+@\S+)@\S+\.(\S+)\.(\S+)\.peaks$",
                    os.path.basename(intervals_file))
    prefix = m.group(1) + "@" + m.group(2)

    # Get DataFrame
    df = pd.read_table(intervals_file, header=0)
    df["START"] = df["abs_summit"] - 100 - 1
    df["END"] = df["abs_summit"] + 100

    # Get BED file
    intervals = []
    bed_file = os.path.join(dummy_dir,
                            "%s+%s+%s.bed" % (os.path.split(__file__)[1],
                                              str(os.getpid()), prefix))
    for _, row in df.iterrows():
        i = [row[0], row[1], row[2], row[8], row[4], "."]
        intervals.append("\t".join(map(str, i)))
    a = BedTool("\n".join([i for i in intervals]), from_string=True)
    a.saveas(bed_file)

    # To ExplaiNN
    _to_ExplaiNN([bed_file], genome_file, dummy_dir="/tmp/",
                 output_dir=output_dir, prefix=prefix, random_seed=1714,
                 splits=[80, 20, 0])

    # Remove BED file
    os.remove(bed_file)

    # Create set for PWM scoring
    validation_file = os.path.join(output_dir, f"{prefix}.validation.tsv.gz")
    test_file = os.path.join(output_dir, f"{prefix}.pwm-scoring.tsv.gz")
    ones = pd.read_table(validation_file, header=None)
    zeros = ones.copy(deep=True)
    ones[2] = 1.
    zeros[0] += "_shuffled"
    zeros[1] = [shuffle_string(s, k=2, random_seed=1714) for s in ones[1].tolist()]
    zeros[2] = 0.
    df = pd.concat((ones, zeros)).reset_index(drop=True)
    df.to_csv(test_file, sep="\t", header=False, index=False,
              compression="gzip")

if __name__ == "__main__":
    main()