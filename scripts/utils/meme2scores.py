#!/usr/bin/env python

import click
from click_option_group import optgroup
import json
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch
import torch.nn as nn
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

class PWM(nn.Module):
    """PWM (Position Weight Matrix)."""

    def __init__(self, pwms, input_length, scoring="max"):
        """
        initialize the model

        :param pwms: arr, numpy array with shape = (n, 4, PWM length)
        :param input_length: int, input sequence length
        :param scoring: string, return either the max. score or the sum
                        occupancy score for each sequence, default max
        """
        super(PWM, self).__init__()

        num_pwms, _, filter_size = pwms.shape

        self._options = {
            "num_pwms": num_pwms,
            "filter_size": filter_size,
            "input_length": input_length,
            "scoring": scoring
        }

        self.conv1d = nn.Conv1d(in_channels=4 * num_pwms, out_channels=1 * num_pwms, kernel_size=filter_size,
                                groups=num_pwms)

        self.conv1d.bias.data = torch.Tensor([0.] * num_pwms) # no bias

        self.conv1d.weight.data = torch.Tensor(pwms) # set the conv. weights
                                                     # to the PWM weights

        for p in self.conv1d.parameters():
            p.requires_grad = False  # freeze

    def forward(self, x):
        """Forward propagation of a batch."""
        x_rev = _flip(_flip(x, 1), 2)
        o = self.conv1d(x.repeat(1, self._options["num_pwms"], 1))
        o_rev = self.conv1d(x_rev.repeat(1, self._options["num_pwms"], 1))
        o = torch.cat((o, o_rev), 2)
        if self._options["scoring"] == "max":
            return torch.max(o, 2)[0]
        else:
            return torch.sum(o, 2)

def _flip(x, dim):
    """
    Reverses the elements in a given dimension `dim` of the Tensor.
    source: https://github.com/pytorch/pytorch/issues/229
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
        x.size(0), x.size(1), -1)[:, getattr(
            torch.arange(x.size(1)-1, -1, -1),
            ("cpu","cuda")[x.is_cuda])().long(), :]

    return x.view(xsize)

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
    "-c", "--cpu-threads",
    help="Number of CPU threads to use.",
    type=int,
    default=1,
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
    "-t", "--time",
    help="Return the program's running execution time in seconds.",
    is_flag=True,
)
@optgroup.group("PWM scoring")
@optgroup.option(
    "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--input-length",
    help="Input length (for longer and shorter sequences, trim or add padding, i.e. Ns, up to the specified length).",
    type=int,
    required=True,
)
@click.option(
    "-s", "--scoring",
    help="Scoring function.",
    type=click.Choice(["max", "sum"]),
    default="max",
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

    ##############
    # Load Data  #
    ##############

    # Get training sequences and labels
    seqs, labels, _ = get_seqs_labels_ids(args["tsv_file"],
                                            args["debugging"],
                                            False,
                                            args["input_length"])

    # Get DataLoader
    data_loader = get_data_loader(seqs, labels, args["batch_size"])

    # Load PWMs and names
    pwms, names = _get_PWMs(args["meme_file"], resize_pwms=True,
        return_log=True)

    ##############
    # Score PWMs #
    ##############

    # Initialize
    idx = 0
    scores = np.zeros((len(seqs), len(pwms)))

    # Infer input length/type, and the number of classes
    input_length = seqs[0].shape[1]

    # Get device
    device = get_device()

    # Get model
    m = PWM(pwms, input_length, args["scoring"]).to(device)

    with torch.no_grad():
        for x, _ in tqdm(iter(data_loader), total=len(data_loader),
                         bar_format=bar_format):

            x = x.to(device) # prepare inputs

            s = m(x) # get scores
            scores[idx:idx+x.shape[0], :] = s.cpu().numpy()

            idx += x.shape[0] # increase index

    ###############
    # AUC metrics #
    ###############

    # Initialize
    aucs = []
    metrics = dict(aucROC=roc_auc_score, aucPR=average_precision_score)

    # Compute AUCs
    for i in range(len(names)):
        s = scores[:, i]
        aucs.append([names[i]])
        for m in metrics:
            aucs[-1].append(metrics[m](labels, s))

    ###############
    # Output AUCs #
    ###############

    tsv_file = os.path.join(args["output_dir"], "scores.tsv")
    if not os.path.exists(tsv_file):
        df = pd.DataFrame(aucs, columns=["PWM"]+[m for m in metrics])
        df.to_csv(tsv_file, sep="\t", index=False)

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f'Execution time {seconds} seconds')

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
            m = re.search("^[\t\s]*(\S+)[\t\s]+(\S+)[\t\s]+(\S+)[\t\s]+(\S+)", line)
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
        if return_log:
            pwm = np.log(pwm)
        pwms.append(np.array(pwm))

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