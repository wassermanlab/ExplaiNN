#!/usr/bin/env python

import click
from click_option_group import optgroup
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch

from explainn.train.train import train_explainn
from explainn.utils.tools import pearson_loss
from explainn.models.networks import ExplaiNN
from utils import (get_file_handle, get_seqs_labels_ids, get_data_loader,
                   get_device)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "training_file",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "validation_file",
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
@optgroup.group("ExplaiNN")
@optgroup.option(
    "--filter-size",
    help="Size of each unit's filter.",
    type=int,
    default=19,
    show_default=True,
)
@optgroup.option(
    "--input-length",
    help="Input length (for longer and shorter sequences, trim or add padding, i.e. Ns, up to the specified length).",
    type=int,
    required=True,
)
@optgroup.option(
    "--num-fc",
    help="Number of fully connected layers in each unit.",
    type=click.IntRange(0, 8, clamp=True),
    default=2,
    show_default=True,
)
@optgroup.option(
    "--num-units",
    help="Number of independent units.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--pool-size",
    help="Size of each unit's maxpooling layer.",
    type=int,
    default=7,
    show_default=True,
)
@optgroup.option(
    "--pool-stride",
    help="Stride of each unit's maxpooling layer.",
    type=int,
    default=7,
    show_default=True,
)
@optgroup.group("Optimizer")
@optgroup.option(
    "--criterion",
    help="Loss (objective) function to use. Select \"BCEWithLogits\" for binary or multi-class classification tasks (e.g. predict the binding of one or more TFs to a sequence), \"CrossEntropy\" for multi-class classification tasks wherein only one solution is possible (e.g. predict the species of origin of a sequence between human, mouse or zebrafish), \"MSE\" for regression tasks (e.g. predict probe intensity signals), \"Pearson\" also for regression tasks (e.g. modeling accessibility across 81 cell types), and \"PoissonNLL\" for modeling count data (e.g. total number of reads at ChIP-/ATAC-seq peaks).",
    type=click.Choice(["BCEWithLogits", "CrossEntropy", "MSE", "Pearson", "PoissonNLL"], case_sensitive=False),
    required=True
)
@optgroup.option(
    "--lr",
    help="Learning rate.",
    type=float,
    default=0.0005,
    show_default=True,
)
@optgroup.option(
    "--optimizer",
    help="`torch.optim.Optimizer` with which to minimize the loss during training.",
    type=click.Choice(["Adam", "SGD"], case_sensitive=False),
    default="Adam",
    show_default=True,
)
@optgroup.group("Training")
@optgroup.option(
    "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--checkpoint",
    help="How often to save checkpoints (e.g. 1 means that the model will be saved after each epoch; by default, i.e. 0, only the best model will be saved).",
    type=int,
    default=0,
    show_default=True,
)
@optgroup.option(
    "--num-epochs",
    help="Number of epochs to train the model.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--patience",
    help="Number of epochs to wait before stopping training if the validation loss does not improve.",
    type=int,
    default=10,
    show_default=True,
)
@optgroup.option(
    "--rev-complement",
    help="Reverse and complement training sequences.",
    is_flag=True,
)
@optgroup.option(
    "--trim-weights",
    help="Constrain output weights to be non-negative (i.e. to ease interpretation).",
    is_flag=True,
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

    # Get training/test sequences and labels
    train_seqs, train_labels, _ = get_seqs_labels_ids(args["training_file"],
                                                      args["debugging"],
                                                      args["rev_complement"],
                                                      args["input_length"])
    test_seqs, test_labels, _ = get_seqs_labels_ids(args["validation_file"],
                                                    args["debugging"],
                                                    args["rev_complement"],
                                                    args["input_length"])

    # Get training/test DataLoaders
    train_loader = get_data_loader(train_seqs, train_labels,
                                   args["batch_size"], shuffle=True)
    test_loader = get_data_loader(test_seqs, test_labels,
                                  args["batch_size"], shuffle=True)

    ##############
    # Train      #
    ##############

    # Infer input length/type, and the number of classes
    # input_length = train_seqs[0].shape[1]
    num_classes = train_labels[0].shape[0]

    # Get device
    device = get_device()

    # Get criterion
    if args["criterion"].lower() == "bcewithlogits":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args["criterion"].lower() == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif args["criterion"].lower() == "mse":
        criterion = torch.nn.MSELoss()
    elif args["criterion"].lower() == "pearson":
        criterion = pearson_loss
    elif args["criterion"].lower() == "poissonnll":
        criterion = torch.nn.PoissonNLLLoss()

    # Get model and optimizer
    m = ExplaiNN(args["num_units"], args["input_length"], num_classes,
                 args["filter_size"], args["num_fc"], args["pool_size"],
                 args["pool_stride"])

    # Get optimizer
    o = _get_optimizer(args["optimizer"], m.parameters(), args["lr"])

    # Train
    _train(train_loader, test_loader, m, device, criterion, o,
           args["num_epochs"], args["output_dir"], None, True, False,
           args["checkpoint"], args["patience"])

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Execution time {seconds} seconds")

def _get_optimizer(optimizer, parameters, lr=0.0005):

    if optimizer.lower() == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=lr)

def _train(train_loader, test_loader, model, device, criterion, optimizer,
           num_epochs=100, output_dir="./", name_ind=None, verbose=False,
           trim_weights=False, checkpoint=0, patience=0):

    # Initialize
    model.to(device)

    # Train
    _, train_error, test_error = train_explainn(train_loader, test_loader,
                                                model, device, criterion,
                                                optimizer, num_epochs,
                                                output_dir, name_ind,
                                                verbose, trim_weights,
                                                checkpoint, patience)

    # Save losses
    df = pd.DataFrame(list(zip(train_error, test_error)),
                      columns=["Train loss", "Validation loss"])
    df.index += 1
    df.index.rename("Epoch", inplace=True)
    df.to_csv(os.path.join(output_dir, "losses.tsv"), sep="\t")

if __name__ == "__main__":
    cli()
