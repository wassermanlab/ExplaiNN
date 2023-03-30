#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import os
import re
import subprocess as sp
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from fastq2explainn import _to_ExplaiNN

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
    splits = []
    prefixes = []

    # For each read file...
    for i, reads_file in enumerate(sorted(reads_files)):

        # Get data splits, prefixes
        m = re.search("^(\S+@\S+@\S+)\.C\d\.5\w+\.3\w+@\S+\.(\S+)\.(\S+)\.fastq.gz$",
                      os.path.basename(reads_file))
        if prefix is None:
            prefix = m.group(1)
            if m.group(3) == "Train":
                splits = [100, 0, 0]
            elif m.group(3) == "Val":
                splits = [0, 100, 0]
        prefixes.append(m.group(2))

    # To ExplaiNN
    prefix += "@%s" % "+".join(prefixes)
    _to_ExplaiNN(clip_left=None, clip_right=None, dummy_dir="/tmp/",
                 fastq_1=reads_files, fastq_2=[], output_dir=output_dir,
                 prefix=prefix, random_seed=1714, splits=splits)

if __name__ == "__main__":
    main()