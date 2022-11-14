#!/usr/bin/env python

from Bio import SeqIO
from Bio import motifs
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter
import click
import gzip
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

# Local imports
from architectures import ExplaiNN, get_metrics
from jaspar import get_figure, reformat_motif
from sequence import rc_many
from train import _get_seqs_labels_ids, _get_data_loader

# Globals
activations = None
outputs = None
predictions = None

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "model_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "test_file",
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
    "-r", "--rev-complement",
    help="Reverse complement sequences.",
    is_flag=True,
)

def main(**args):

    # Globals
    global activations
    global outputs
    global predictions

    ##############
    # Load Data  #
    ##############

    # Get data
    seqs, labels, _ = _get_seqs_labels_ids(args["test_file"],
        args["debugging"], args["rev_complement"])

    # Get DataLoader
    data_loader = _get_data_loader(seqs, labels, args["batch_size"])

    # Load model
    exp_model = _load_model(args["model_file"])

    ##############
    # Test       #
    ############## 

    # Initialize
    if np.unique(labels[:, 0]).size == 2:
        input_data = "binary"
    else:
        input_data = "linear"

    # Create output dirs
    if not os.path.isdir(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # Compute activations, outputs, and predictions
    _compute_acts_outs_preds(exp_model, data_loader, args["output_dir"])

    # Save predictions
    predictions_file = os.path.join(args["output_dir"], "predictions.npz")
    np.savez_compressed(predictions_file, predictions)

    # Get performance metrics
    metrics = get_metrics(input_data=input_data)
    tsv_file = os.path.join(args["output_dir"], "performance-metrics.tsv")
    if not os.path.exists(tsv_file):
        data = []
        for m in metrics:
            p = _get_performances(predictions, labels, input_data, metrics[m],
                args["rev_complement"])
            data.append([m] + p)
        column_names = ["metric", "global"] + list(range(labels.shape[1]))
        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(tsv_file, sep="\t", index=False)

def _load_model(model_file):

    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model
    selene_dict = torch.load(model_file)
    exp_model = ExplaiNN(
        selene_dict["options"]["cnn_units"],
        selene_dict["options"]["kernel_size"],
        selene_dict["options"]["sequence_length"],
        selene_dict["options"]["n_features"],
        selene_dict["options"]["weights_file"],
    )
    exp_model.load_state_dict(selene_dict["state_dict"])
    exp_model.to(device)
    exp_model.eval()

    return(exp_model)

def _compute_acts_outs_preds(exp_model, data_loader, output_dir="./"):

    # Initialize
    idx = 0
    x = len(data_loader.dataset)
    y = exp_model._options["cnn_units"]
    z = exp_model._options["sequence_length"] - \
        exp_model._options["kernel_size"] + 1
    n_features = exp_model._options["n_features"]
    global activations
    activations = np.zeros((x, y, z), dtype=np.float16)
    global outputs
    outputs = np.zeros((x, y), dtype=np.float16)
    global predictions
    predictions = np.zeros((x, n_features), dtype=np.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for Xs, _ in tqdm(iter(data_loader), total=len(data_loader),
                bar_format=bar_format):

            # Prepare input
            Xs = Xs.to(device)
            Xs = Xs.repeat(1, exp_model._options["cnn_units"], 1)

            # Get outputs
            outs = exp_model.linears(Xs)
            outputs[idx:idx+Xs.shape[0], :] = outs.cpu().numpy()

            # Get predictions
            preds = exp_model.final(outs)
            predictions[idx:idx+Xs.shape[0]] = preds.cpu().numpy()

            # Get activations
            activations[idx:idx+Xs.shape[0], :, :] = \
                exp_model.linears[:3](Xs).cpu().numpy()

            # Index increase
            idx += Xs.shape[0]

def _get_performances(predictions, labels, input_data, metric,
                      rev_complement=False):

    # Initialize
    performances = []

    if rev_complement:
        fwd = __get_fwd_rev(predictions, "fwd")
        rev = __get_fwd_rev(predictions, "rev")
        p = np.empty(fwd.shape)
        ys = __get_fwd_rev(labels, "fwd")
        # Average predictions from forward and reverse strands
        for i in range(p.shape[1]):
            p[:, i] = np.mean([fwd[:, i], rev[:, i]], axis=0)
            if input_data == "binary":
                p[:, i] = torch.sigmoid(torch.from_numpy(p[:, i])).numpy()
    else:
        if input_data == "binary":
            p = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        else:
            p = predictions
        ys = labels

    # For each class...
    performances.append(metric(ys, p))
    for i in range(ys.shape[1]):
        performances.append(metric(ys[:, i], p[:, i]))

    return(performances)

def __get_fwd_rev(arr, strand):

    if strand == "fwd" or strand == "+":
        return(arr[:len(arr)//2])
    elif strand == "rev" or strand == "-":
        return(arr[len(arr)//2:])

if __name__ == "__main__":
    main()