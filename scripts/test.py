#!/usr/bin/env python

import click
from click_option_group import optgroup
import json
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch

from explainn.interpretation.interpretation import get_explainn_predictions
from explainn.models.networks import ExplaiNN
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "model_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "training_parameters_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "test_file",
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
@optgroup.group("Test")
@optgroup.option(
    "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)

def cli(**args):

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

    # Load training parameters
    handle = get_file_handle(args["training_parameters_file"], "rt")
    train_args = json.load(handle)
    handle.close()

    # Get training sequences and labels
    seqs, labels, _ = get_seqs_labels_ids(args["test_file"],
                                          args["debugging"],
                                          False,
                                          train_args["input_length"])

    ##############
    # Test       #
    ############## 

    # Infer input type, and the number of classes
    num_classes = labels[0].shape[0]
    if np.unique(labels[:, 0]).size == 2:
        input_type = "binary"
    else:
        input_type = "non-binary"

    # Get device
    device = get_device()

    # Get model
    m = ExplaiNN(train_args["num_units"], train_args["input_length"],
                 num_classes, train_args["filter_size"], train_args["num_fc"],
                 train_args["pool_size"], train_args["pool_stride"],
                 train_args["weights_file"])
    m.load_state_dict(torch.load(args["model_file"]))

    # Test
    _test(seqs, labels, m, device, input_type, train_args["filter_size"],
          train_args["rev_complement"], args["output_dir"], args["batch_size"])

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Execution time {seconds} seconds")

def _test(seqs, labels, model, device, input_type, filter_size,
          rev_complement, output_dir="./", batch_size=100):

    # Initialize
    predictions = []
    model.to(device)
    model.eval()

    # Get training DataLoader
    data_loader = get_data_loader(seqs, labels, batch_size)

    # Get rev. complement
    if rev_complement:
        rev_seqs = np.array([s[::-1, ::-1] for s in seqs])
        rev_data_loader = get_data_loader(rev_seqs, labels, batch_size)
    else:
        rev_seqs = None
        rev_data_loader = None

    for dl in [data_loader, rev_data_loader]:

        # Skip
        if dl is None:
            continue

        # Get predictions
        preds, labels = get_explainn_predictions(dl, model, device,
                                                 isSigmoid=False)
        predictions.append(preds)

    # Avg. predictions from both strands
    if len(predictions) == 2:
        avg_predictions = np.empty(predictions[0].shape)
        for i in range(predictions[0].shape[1]):
            avg_predictions[:, i] = np.mean([predictions[0][:, i],
                                            predictions[1][:, i]], axis=0)
    else:
        avg_predictions = predictions[0]
    if input_type == "binary":
        for i in range(avg_predictions.shape[1]):
            avg_predictions[:, i] = \
                torch.sigmoid(torch.from_numpy(avg_predictions[:, i])).numpy()

    # Get performance metrics
    metrics = __get_metrics(input_data=input_type)
    tsv_file = os.path.join(output_dir, "performance-metrics.tsv")
    if not os.path.exists(tsv_file):
        data = []
        for m in metrics:
            data.append([m])
            data[-1].append(metrics[m](labels, avg_predictions))
            for i in range(labels.shape[1]):
                data[-1].append(metrics[m](labels[:, i],
                                           avg_predictions[:, i]))
        column_names = ["metric", "global"] + list(range(labels.shape[1]))
        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(tsv_file, sep="\t", index=False)

def __get_metrics(input_data="binary"):

    if input_data == "binary":
        return(dict(aucROC=roc_auc_score, aucPR=average_precision_score))

    return(dict(Pearson=pearsonr, Spearman=spearmanr))

if __name__ == "__main__":
    cli()