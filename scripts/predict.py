#!/usr/bin/env python

from Bio import SeqIO
import click
from io import StringIO
import numpy as np
import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

# Local imports
from architectures import ExplaiNN
from sequence import one_hot_encode_many, rc_one_hot_encoding_many
from utils import get_file_handle

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "model_file",
    type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "fasta_file",
    type=click.Path(exists=True, resolve_path=True)
)
@click.option(
    "-b", "--batch-size",
    help="Batch size.",
    type=int,
    default=2**6,
    show_default=True,
)
@click.option(
    "-o", "--output-file",
    help="Output file.  [default: stdout]",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-s", "--apply-sigmoid",
    help="Apply the logistic sigmoid function to outputs.",
    is_flag=True,
)

def main(**args):

    ##############
    # Load Data  #
    ##############

    # Get data
    Xs, seq_ids = _get_Xs_ids(args["fasta_file"])

    # Get DataLoader
    data_loader = DataLoader(TensorDataset(torch.Tensor(Xs),
        torch.Tensor(rc_one_hot_encoding_many(Xs))), args["batch_size"])

    # Load model
    model = _load_model(args["model_file"])

    ##############
    # Predict    #
    ############## 

    # Initialize
    idx = 0
    predictions = np.empty((len(Xs), model._options["n_features"], 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():

        for i, (fwd, rev) in tqdm(enumerate(iter(data_loader)),
            total=len(data_loader), bar_format=bar_format):

            # Get strand-specific predictions
            fwd = np.expand_dims(model(fwd.to(device)).cpu().numpy(), axis=2)
            rev = np.expand_dims(model(rev.to(device)).cpu().numpy(), axis=2)

            # Combine predictions from both strands
            fwd_rev = np.concatenate((fwd, rev), axis=2)
            mean_fwd_rev = np.expand_dims(np.mean(fwd_rev, axis=2), axis=2)
            max_fwd_rev = np.expand_dims(np.max(fwd_rev, axis=2), axis=2)

            # Concatenate predictions for this batch
            p = np.concatenate((fwd, rev, mean_fwd_rev, max_fwd_rev), axis=2)
            predictions[idx:idx+fwd.shape[0]] = p

            # Index increase
            idx += fwd.shape[0]

    # Apply sigmoid
    if args["apply_sigmoid"]:
        predictions = torch.sigmoid(torch.Tensor(predictions)).numpy()

    ##############
    # Output     #
    ############## 

    dfs = []
    for i in range(model._options["n_features"]):
        p = predictions[:, i, :]
        df = pd.DataFrame(p, columns=["Fwd", "Rev", "Mean", "Max"])
        df["SeqId"] = seq_ids
        df["Class"] = i 
        dfs.append(df)
    df = pd.concat(dfs)[["SeqId", "Class", "Fwd", "Rev", "Mean", "Max"]]
    df.reset_index(drop=True, inplace=True)
    if args["output_file"] is not None:
        df.to_csv(args["output_file"], sep="\t", index=False)
    else:
        o = StringIO()
        df.to_csv(o, sep="\t", index=False)
        sys.stdout.write(o.getvalue())

def _get_Xs_ids(fasta_file):

    # Get sequences
    fh = get_file_handle(fasta_file, "rt")
    records = list(SeqIO.parse(fh, "fasta"))
    fh.close()

    # Xs / ids
    Xs = one_hot_encode_many(np.array([str(r.seq) for r in records]))
    ids = np.array([r.id for r in records])

    return(Xs, ids)

def _load_model(model_file):

    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model
    selene_dict = torch.load(model_file)
    model = ExplaiNN(
        selene_dict["options"]["cnn_units"],
        selene_dict["options"]["kernel_size"],
        selene_dict["options"]["sequence_length"],
        selene_dict["options"]["n_features"],
        selene_dict["options"]["weights_file"],
    )
    model.load_state_dict(selene_dict["state_dict"])
    model.to(device)
    model.eval()

    return(model)

if __name__ == "__main__":
    main()
