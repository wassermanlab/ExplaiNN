#!/usr/bin/env python

import click
import numpy as np
import os
import pandas as pd
import re
import sys
import torch
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from explainn.models.networks import PWM

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), os.pardir))
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "meme_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "tsv_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-b", "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    "-d", "--debugging",
    help="Debugging mode.",
    is_flag=True,
)
@click.option(
    "-o", "--output-dir",
    help="Output directory.",
    type=click.Path(resolve_path=True),
    default="./",
    show_default=True,
)
@click.option(
    "-p", "--prefix",
    help="Output prefix.",
)
@click.option(
    "-s", "--scoring",
    help="Scoring function.",
    type=click.Choice(["max", "sum"]),
    default="max",
    show_default=True,
)

def main(**args):

    # Create output dir
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    ##############
    # Load Data  #
    ##############

    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get data
    seqs, y_true, _ = _get_seqs_labels_ids(args["tsv_file"], args["debugging"])

    # Get DataLoader
    data_loader = _get_data_loader(seqs, y_true, args["batch_size"])

    # Load model
    pwms, names = _get_PWMs(args["meme_file"], resize_pwms=True,
        return_log=True)
    pwm_model = PWM(pwms, seqs.shape[2], args["scoring"]).to(device)

    ##############
    # Score PWMs #
    ##############

    # Initialize
    idx = 0
    scores = np.zeros((len(data_loader.dataset), pwm_model._options["groups"]))

    with torch.no_grad():
        for x, _ in tqdm(iter(data_loader), total=len(data_loader),
                bar_format=bar_format):

            # Prepare inputs
            x = x.to(device)

            # Get scores
            s = pwm_model(x)
            scores[idx:idx+x.shape[0], :] = s.cpu().numpy()

            # Index increase
            idx += x.shape[0]

    ###############
    # AUC metrics #
    ###############

    # Initialize
    aucs = []
    metrics = get_metrics()

    # Compute AUCs
    for i in range(len(names)):
        y_score = scores[:, i]
        aucs.append([names[i]])
        for m in metrics:
            aucs[-1].append(metrics[m](y_true, y_score))

    ###############
    # Output AUCs #
    ###############

    # Create DataFrame
    df = pd.DataFrame(aucs, columns=["PWM"]+[m for m in metrics])

    # Save AUCs
    if args["prefix"] is None:
        tsv_file = os.path.join(args["output_dir"], "%s.tsv" % args["scoring"])
    else:
        tsv_file = os.path.join(args["output_dir"],
            "%s.%s.tsv" % (args["prefix"], args["scoring"]))
    df.to_csv(tsv_file, sep="\t", index=False)

def _get_PWMs(meme_file, resize_pwms=False, return_log=False):

    # Initialize
    dicts = []
    names = []
    pwms = []
    alphabet = "ACGT"
    parse = False

    # Get PWM
    handle = get_file_handle(meme_file, "rt")
    for line in handle:
        line = line.strip("\n")
        if line.startswith("MOTIF"):
            parse = True
            dicts.append({})
            for l in alphabet:
                dicts[-1].setdefault(l, [])
            names.append(line.split(" ")[1])
        elif not parse:
            continue
        elif line.startswith("letter-probability matrix:"):
            continue
        else:
            m = re.search("^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", line)
            if m:
                for l in range(len(alphabet)):
                    # Add pseudocounts
                    v = max([1e-4, float(m.group(l+1))])
                    dicts[-1][alphabet[l]].append(v)

    # Get max. PWM size
    max_size = 0
    for d in dicts:
        for l in alphabet:
            max_size = max([len(d[l]), max_size])
            break

    # For each matrix...
    for d in dicts:
        pwm = []
        for l in alphabet:
            pwm.append(d[l])
        if resize_pwms:
            pwm = __resize_PWM(list(zip(*pwm)), max_size)
            pwm = list(zip(*pwm))
        pwms.append(pwm)

    if return_log:
        return(np.log(pwms), names)
    else:
        return(pwms, names)

def __resize_PWM(pwm, size):

    # Initialize
    lpop = 0
    rpop = 0

    pwm = [[.25,.25,.25,.25]]*size+pwm+[[.25,.25,.25,.25]]*size

    while len(pwm) > size:
        if max(pwm[0]) < max(pwm[-1]):
            pwm.pop(0)
            lpop += 1
        elif max(pwm[-1]) < max(pwm[0]):
            pwm.pop(-1)
            rpop += 1
        else:
            if lpop > rpop:
                pwm.pop(-1)
                rpop += 1
            else:
                pwm.pop(0)
                lpop += 1

    return(pwm)

if __name__ == "__main__":
    main()