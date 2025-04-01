#!/usr/bin/env python

import os
import sys
import time
import torch
import click
import json

import logging
logging.basicConfig(
    format="{asctime} - {name} - {levelname} - {message}", 
    style="{",
    datefmt="%Y-%m-%d %H:%M", 
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))

from explainn.models.networks import ExplaiNN
from explainn.interpretation.interpretation import get_explainn_predictions
from run.utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device, data_split_names, get_criterion, validate_config)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}
@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "config_file",
    type=click.Path(exists=True, resolve_path=True),
)
def main(**args):
    """
    """
    # Read config file
    with open(args["config_file"]) as f:
        config = json.load(f)
        
    # Validate the fields of the config file
    try:
        validate_config(config)
        logging.info("Config file validated.")
    except Exception as e:
        logging.error(str(e))

    # Check that output dir exists
    output_dir = config["data"]["output_dir"]
    if not os.path.isdir(output_dir):
        raise OSError(
            f"The output directory: {output_dir} does not exist.\n"
            f"Check the path relative to the current working directory: {os.getcwd()}"
        )
    
    test_model(config)


def test_model(config):
    """
    """
    # Start execution
    start_time = time.time()

    # Get input data
    inputs = data_split_names(config["data"]["output_dir"], config["data"]["prefix"])

    # Get test sequences and labels
    seqs, labels, _ = get_seqs_labels_ids(
        inputs["test"],
        config["options"]["debugging"],
        False, 
        int(config["data"]["input_length"])
    )

    # Infer input type, and the number of classes
    num_classes = labels[0].shape[0]
    if np.unique(labels[:, 0]).size == 2:
        input_type = "binary"
    else:
        input_type = "non-binary"

    device = get_device()

    # TODO: find a better way of getting model file from output of train
    for i,file in enumerate(os.listdir(config["data"]["output_dir"])):
        if file.endswith(".pth"):
            model_file = os.path.join(config["data"]["output_dir"], file)

    # Get model
    m = ExplaiNN(config["cnn"]["num_units"], config["data"]["input_length"], 
            num_classes, config["cnn"]["filter_size"], config["cnn"]["num_fc"], 
            config["cnn"]["pool_size"], config["cnn"]["pool_stride"],
            model_file)

    # Test
    _test(seqs, labels, m, device, input_type, config["data"]["rev_complement"],
        config["data"]["output_dir"], config["training"]["batch_size"])

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if config["options"]["use_time"]:
        f = os.path.join(config["data"]["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Test execution time {seconds} seconds")


def _test(seqs, labels, model, device, input_type, rev_complement,
          output_dir="./", batch_size=100):

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
        column_names = ["metric"]
        for m in metrics:
            data.append([m])
            if labels.shape[1] > 1:
                data[-1].append(metrics[m](labels, avg_predictions))
                column_names = column_names + ["global"]
            for i in range(labels.shape[1]):
                data[-1].append(metrics[m](labels[:, i],
                                           avg_predictions[:, i]))
        if labels.shape[1] > 1:
            column_names = ["metric", "global"] + list(range(labels.shape[1]))
        else:
            column_names = ["metric"] + list(range(labels.shape[1]))
        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(tsv_file, sep="\t", index=False)

def __get_metrics(input_data="binary"):

    if input_data == "binary":
        return(dict(aucROC=roc_auc_score, aucPR=average_precision_score))

    return(dict(Pearson=pearson_corrcoef))

def pearson_corrcoef(y_true, y_score):

    if y_true.ndim == 1:
        return np.corrcoef(y_true, y_score)[0, 1]
    else:
        if y_true.shape[1] == 1:
            return np.corrcoef(y_true, y_score)[0, 1]
        else:
            corrcoefs = []
            for i in range(len(y_score)):
                x = np.corrcoef(y_true[i, :], y_score[i, :])[0, 1]
                corrcoefs.append(x)
            return np.mean(corrcoefs)

if __name__=="__main__":
    main()