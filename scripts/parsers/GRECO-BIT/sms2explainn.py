#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import os
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

    # Get ExplaiNN files
    kwargs = {"total": len(reads_files), "bar_format": bar_format}
    pool = Pool(args["cpu_threads"])
    p = partial(_get_ExplaiNN_files, output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, reads_files), **kwargs):
        pass

def _get_ExplaiNN_files(reads_file, output_dir="./"):
    """
    Naming format:
    AKAP8L@silly-willy+clever-peter@Megaman.HOMER@motif-id123.pcm
    """

    # Initialize
    prefix = None

    # Get prefix
    m = re.search("^(\S+@\S+@\S+).5\w+\.3\w+@\S+\.(\S+)\.\S+\.fastq",
                    os.path.basename(reads_file))
    prefix = m.group(1) + "@" + m.group(2)

    # To ExplaiNN
    _to_ExplaiNN(clip_left=None, clip_right=None, dummy_dir="/tmp/",
                 fastq_1=[reads_file], fastq_2=[], output_dir=output_dir,
                 prefix=prefix, random_seed=1714, splits=[80, 20, 0])

    # Create set for PWM scoring
    validation_file = os.path.join(output_dir, f"{prefix}.validation.tsv.gz")
    test_file = os.path.join(output_dir, f"{prefix}.pwm-scoring.tsv.gz")
    os.symlink(validation_file, test_file)

if __name__ == "__main__":
    main()