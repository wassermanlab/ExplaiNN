#!/usr/bin/env python

import click
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import re
import sys
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from jaspar2logo import _get_figure
from meme2scores import _get_PWMs
from utils import get_file_handle

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "meme_file",
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
    "-f", "--oformat",
    help="Output format.",
    default="png",
    show_default=True,
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True,
)

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Get PWMs
    pwms, names = _get_PWMs(args["meme_file"])

    # Generate logos
    kwargs = {"bar_format": bar_format, "total": len(pwms)}
    pool = Pool(args["cpu_threads"])
    p = partial(_generate_logo, oformat=args["oformat"],
        output_dir=args["output_dir"])
    for _ in tqdm(pool.imap(p, zip(pwms, names)), **kwargs):
        pass

def _generate_logo(pwm_name, oformat="png", output_dir="./"):

    # Initialize
    pwm, name = pwm_name

    for reverse_complement in [False, True]:
        if reverse_complement:
            logo_file = os.path.join(output_dir, f"{name}.rev.{oformat}")
        else:
            logo_file = os.path.join(output_dir, f"{name}.fwd.{oformat}")
        if not os.path.exists(logo_file):
            try:
                fig = _get_figure(pwm, reverse_complement)
                fig.savefig(logo_file, bbox_inches="tight", pad_inches=0)
            except:
                # i.e. no motif
                fh = get_file_handle(logo_file, "wt")
                fh.close()

if __name__ == "__main__":
    main()