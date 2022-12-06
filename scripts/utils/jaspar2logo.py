#!/usr/bin/env python

import click
from Bio import motifs
import logomaker
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Specify font
f = os.path.join(os.path.dirname(os.path.realpath(__file__)), "utils", "fonts",
                 "Arial.ttf")
prop = fm.FontProperties(fname=f)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "jaspar_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "logo_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-r", "--rev-complement",
    help="Plot the reverse complement logo.",
    is_flag=True,
)

def main(**args):

    # Get figure
    fig = get_figure(args["motif_file"], rc=args["rev_complement"])

    # Save
    fig.savefig(args["logo_file"], bbox_inches="tight", pad_inches=0)

def get_figure(motif_file, rc=False):

    # From https://biopython.readthedocs.io/en/latest/chapter_motifs.html
    m = motifs.read(open(motif_file), "jaspar")
    pwm = list(m.counts.normalize(pseudocounts=.5).values())

    return(_get_figure(pwm, rc))

def _get_figure(pwm, rc=False):

    # From https://www.bioconductor.org/packages/release/bioc/html/seqLogo.html
    if rc:
        arr = np.array(pwm)
        pwm = np.flip(arr).tolist()
    IC = 2 + np.add.reduce(pwm * np.log2(pwm))
    df = pd.DataFrame({
        "pos": [i + 1 for i in range(len(IC))],
        "A": pwm[0] * IC,
        "C": pwm[1] * IC,
        "G": pwm[2] * IC,
        "T": pwm[3] * IC
    })
    df = df.set_index("pos")

    # From https://logomaker.readthedocs.io/en/latest/examples.html
    fig, ax = plt.subplots(1, 1, figsize=(len(df)/2.0, 2))
    logo = logomaker.Logo(df, ax=ax, show_spines=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)
    logo.ax.set_aspect(1.5)
    logo.ax.xaxis.set_ticks(list(df.index))
    logo.ax.set_xticklabels(labels=list(df.index), fontproperties=prop)
    logo.ax.set_ylabel("Bits", fontproperties=prop)
    logo.ax.set_ylim(0, 2)
    logo.ax.yaxis.set_ticks([0, 1, 2])
    logo.ax.set_yticklabels(labels=[0, 1, 2], fontproperties=prop)

    return(fig)

if __name__ == "__main__":

    main()