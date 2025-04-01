#!/usr/bin/env python

import logging
import os
import sys
import time
import torch
import click
import json
import constants

import pandas as pd

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))

from explainn.train.train import train_explainn
from explainn.models.networks import ExplaiNN
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
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
    
    run_train(config)


def run_train(config):
    # Start execition
    start_time = time.time()

    # Get input data
    inputs = data_split_names(config["data"]["output_dir"], config["data"]["prefix"])

    # Load data
    train_seqs, train_labels, _ = get_seqs_labels_ids(
        inputs["train"],
        config["options"]["debugging"],
        config["data"]["rev_complement"],
        int(config["data"]["input_length"])
    )
    test_seqs, test_labels, _ = get_seqs_labels_ids(
        inputs["validation"],
        config["options"]["debugging"],
        config["data"]["rev_complement"],
        int(config["data"]["input_length"])
    )

    # Get training/test DataLoaders
    train_loader = get_data_loader(train_seqs, train_labels,
                        int(config["training"]["batch_size"]), shuffle=True
    )
    test_loader = get_data_loader(test_seqs, test_labels,
                        int(config["training"]["batch_size"]), shuffle=True
    )

    # Infer input length/type and the number of classes
    # input_length = train_seqs[0].shape[1]
    num_classes = train_labels[0].shape[0]
    device = get_device()

    # Get criterion
    try:
        criterion = get_criterion()[config["optimizer"]["criterion"].lower()]
    except KeyError:
        raise KeyError(
            f"Invalid criterion '{config['optimizer']['criterion']}'. "
            f"Please choose one of: {', '.join(get_criterion().keys())}"
        )
    

    # Get model
    m = ExplaiNN(config["cnn"]["num_units"], config["data"]["input_length"], 
            num_classes, config["cnn"]["filter_size"], config["cnn"]["num_fc"], 
            config["cnn"]["pool_size"],config["cnn"]["pool_stride"])

    # Get optimizer
    o = _get_optimizer(config["optimizer"]["optimizer"], m.parameters(), 
            config["optimizer"]["lr"])

    # Train
    _train(train_loader, test_loader, m, device, criterion, o,
            config["training"]["num_epochs"], config["data"]["output_dir"], 
            None, True, False, config["training"]["checkpoint"],
            config["training"]["patience"]
    )

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if config["options"]["use_time"]:
        f = os.path.join(config["data"]["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Train execution time {seconds} seconds")


def _get_optimizer(optimizer, parameters, lr=0.0005):
    """
    """
    return constants.OPTIMIZERS[optimizer.lower()](parameters, lr=lr)
    
def _train(train_loader, test_loader, model, device, criterion, optimizer,
    num_epochs=100, output_dir="./", name_ind=None, verbose=False,
    trim_weights=False, checkpoint=0, patience=0):
    """
    """

    # Initialize
    model.to(device)

    # Train
    _, train_error, test_error = train_explainn(train_loader, test_loader,
                                                model, device, criterion,
                                                optimizer, num_epochs,
                                                output_dir, name_ind,
                                                verbose, trim_weights,
                                                checkpoint, patience

    )

    # Save losses
    df = pd.DataFrame(list(zip(train_error, test_error)),
                      columns=["Train loss", "Validation loss"])
    df.index += 1
    df.index.rename("Epoch", inplace=True)
    df.to_csv(os.path.join(output_dir, "losses.tsv"), sep="\t")

    

if __name__=='__main__':
    main()