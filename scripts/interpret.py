#!/usr/bin/env python

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup
import copy                                   
import json
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),
                                os.pardir))
import time
import torch
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

from explainn.models.networks import ExplaiNN
from explainn.interpretation.interpretation import (get_explainn_predictions,
                                                    get_explainn_unit_activations,
                                                    get_explainn_unit_outputs,
                                                    get_specific_unit_importance,
                                                    get_pwms_explainn,
                                                    pwm_to_meme)

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
    "training_file",
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
@optgroup.group("Interpretation")
@optgroup.option(
    "--batch-size",
    help="Batch size.",
    type=int,
    default=100,
    show_default=True,
)
@optgroup.option(
    "--num-well-pred-seqs",
    help="Number of well-predicted sequences to use.  [default: use all sequences]",
    type=int,
)
@optgroup.group("\n  Define \"well-predicted\" sequences based on", cls=RequiredMutuallyExclusiveOptionGroup)
@optgroup.option(
    "--correlation",
    help="If the correlation between the predicted and actual sequence values is >x (e.g. multi-class regression tasks).",
    type=click.FloatRange(0, 1, clamp=True),
)
@optgroup.option(
    "--exact-match",
    help="If the predicted and actual sequence values are equal (e.g. binary or multi-class classification tasks).",
    is_flag=True
)
@optgroup.option(
    "--percentile-bottom",
    help="If the predicted and actual sequence values are within the bottom x percentile (e.g. single-class regression tasks).",
    type=click.IntRange(1, 100, clamp=True),
)
@optgroup.option(
    "--percentile-top",
    help="If the predicted and actual sequence values are within the top x percentile (e.g. single-class regression tasks).",
    type=click.IntRange(1, 100, clamp=True),
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
    seqs, labels, _ = get_seqs_labels_ids(args["training_file"],
                                          args["debugging"],
                                          False,
                                          train_args["input_length"])

    ##############
    # Interpret  #
    ##############

    # Infer input type, and the number of classes
    num_classes = labels[0].shape[0]
    if np.unique(labels[:, 0]).size == 2:
        input_type = "binary"
    else:
        input_type = "non-binary"

    # Get device
    device = get_device()

    # Get criterion/threshold (if applicable) for well-predicted sequences
    if args["correlation"]:
        criterion = "correlation"
        threshold = args["correlation"]
    elif args["exact_match"]:
        criterion = "exact_match"
        threshold = None
    elif args["percentile_bottom"]:
        criterion = "percentile_bottom"
        threshold = args["percentile_bottom"]
    elif args["percentile_top"]:
        criterion = "percentile_top"
        threshold = args["percentile_top"]

    # Get model
    m = ExplaiNN(train_args["num_units"], train_args["input_length"],
                 num_classes, train_args["filter_size"], train_args["num_fc"],
                 train_args["pool_size"], train_args["pool_stride"],
                 train_args["weights_file"])
    m.load_state_dict(torch.load(args["model_file"]))

    # Interpret
    _interpret(seqs, labels, m, device, input_type, criterion, threshold,
               train_args["filter_size"], train_args["rev_complement"],
               args["output_dir"], args["batch_size"],
               args["num_well_pred_seqs"])

    # Finish execution
    seconds = format(time.time() - start_time, ".2f")
    if args["time"]:
        f = os.path.join(args["output_dir"],
            f"time-{os.path.basename(__file__)}.txt")
        handle = get_file_handle(f, "wt")
        handle.write(f"{seconds} seconds")
        handle.close()
    print(f"Execution time {seconds} seconds")

def one_hot_decode(encoded_seq):
    """Reverts a sequence's one hot encoding."""

    seq = []
    code = list("ACGT")
 
    for i in encoded_seq.transpose(1, 0):
        try:
            seq.append(code[int(np.where(i == 1)[0])])
        except:
            # i.e. N?
            seq.append("N")

    return("".join(seq))

def _interpret(seqs, labels, model, device, input_type, criterion,
               threshold, filter_size, rev_complement, output_dir="./",
               batch_size=100, num_well_pred_seqs=None):

    # Initialize
    activations = []
    predictions = []
    outputs = []
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

    # Get predictions
    for dl in [data_loader, rev_data_loader]:
        if dl is None: # skip
            continue
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

    # Get well-predicted sequences
    if criterion == "correlation":
        correlations = []
        for i in range(len(avg_predictions)):
            x = np.corrcoef(labels[i, :], avg_predictions[i, :])[0, 1]
            correlations.append(x)
        idx = np.argwhere(np.asarray(correlations) > threshold).squeeze()
    elif criterion == "exact_match":
        arr = np.round(avg_predictions) # round predictions
        arr = np.equal(arr, labels)
        idx = np.argwhere(np.sum(arr, axis=1) == labels.shape[1]).squeeze()
    elif criterion == "percentile_bottom":
        threshold = threshold / 100.
        arr_1 = np.argsort(labels.flatten())[:int(max(labels.shape)*threshold)]
        arr_2 = np.argsort(avg_predictions.flatten())[:int(max(avg_predictions.shape)*threshold)]
        idx = np.intersect1d(arr_1, arr_2)
    elif criterion == "percentile_top":
        threshold = threshold / 100.
        arr_1 = np.argsort(-labels.flatten())[:int(max(labels.shape)*threshold)]
        arr_2 = np.argsort(-avg_predictions.flatten())[:int(max(avg_predictions.shape)*threshold)]
        idx = np.intersect1d(arr_1, arr_2)
    if num_well_pred_seqs:
        rng = np.random.default_rng()
        size = min(num_well_pred_seqs, len(idx))
        idx = rng.choice(idx, size=size, replace=False)

    # Get training DataLoader
    seqs = seqs[idx]
    labels = labels[idx]
    data_loader = get_data_loader(seqs, labels, batch_size)

    # Get rev. complement
    if rev_complement:
        rev_seqs = np.array([s[::-1, ::-1] for s in seqs])
        rev_data_loader = get_data_loader(rev_seqs, labels, batch_size)
    else:
        rev_seqs = None
        rev_data_loader = None

    # Get activations
    for dl in [data_loader, rev_data_loader]:
        if dl is None: # skip
            continue       
        acts = get_explainn_unit_activations(dl, model, device)
        activations.append(acts)
    if rev_complement:
        seqs = np.concatenate((seqs, rev_seqs))
        activations = np.concatenate(activations)
    else:
        activations = activations[0]

    # Get MEMEs
    meme_file = os.path.join(output_dir, "filters.meme")
    if not os.path.exists(meme_file):
        pwms = get_pwms_explainn(activations, seqs, filter_size)
        pwm_to_meme(pwms, meme_file)

    # Get final linear layer weights
    weights = model.final.weight.detach().cpu().numpy()
    tsv_file = os.path.join(output_dir, "weights.tsv")
    if not os.path.exists(tsv_file):
        data = []
        for i, weight in enumerate(weights.T):
            data.append([f"filter{i}"] + weight.tolist())
        column_names = ["filter"] + list(range(weights.shape[0]))
        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(tsv_file, sep="\t", index=False)

    # Get unit outputs
    for i, dl in enumerate([data_loader, rev_data_loader]):
        if dl is None: # skip
            continue      
        outs = get_explainn_unit_outputs(dl, model, device)
        outputs.append(outs)
    if rev_complement:
        outputs = np.concatenate(outputs)
    else:
        outputs = outputs[0]

    # Get unit importances
    tsv_file = os.path.join(output_dir, "importances.tsv")
    if not os.path.exists(tsv_file):
        data = []
        target_labels = list(range(weights.shape[0]))
        for i in tqdm(range(weights.shape[1]), total=weights.shape[1],
                      bar_format=bar_format):
            imps = get_specific_unit_importance(activations, model, outputs, i,
                                                target_labels)
            imps = np.array([d for d in imps])
            data.extend([[f"filter{i}"] + j.tolist() for j in imps.T])
        column_names = ["filter"] + target_labels
        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(f"{tsv_file}.gz", sep="\t", index=False, compression="gzip")
        df = df.groupby(["filter"]).median()\
               .sort_values([column_names[-1]], ascending=False)
        df.reset_index(inplace=True)
        df.to_csv(tsv_file, sep="\t", index=False)

if __name__ == "__main__":
    cli()