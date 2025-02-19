import os


import click
import gzip
from functools import partial
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from torch import cuda, Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from explainn.utils.tools import dna_one_hot, pearson_loss


def save_data_splits(output_dir="./", train=None, validation=None, 
    test=None, prefix=None):
    """
    Saves training/validation/testing data to directory specified by
    output_dir, adding prefix to the filenames

    Inputs:
        output_dir      str: path to output directory
        train           pd.df: dataframe with training data
        validation      pd.df: dataframe with validation data
        test            pd.df: dataframe with test data
        prefix          str: text to prepend to filenames

    Outputs:
        None
    """
    # Initialize
    files = [
        ("train", train),
        ("validation", validation),
        ("test", test)
    ]

    # TODO: Check that prefix has no weird characters?
    tsv_paths = data_split_names(output_dir, prefix)

    # Write TSV files
    for i in files:
        # Save TSV if not None
        try:
            i[1].to_csv(tsv_paths[i[0]], sep="\t", header=False, index=False, 
                        compression="gzip")
        except AttributeError:
            continue


def data_split_names(output_dir, prefix=None):
    """
    TODO: Move to constants.py?
    """
    return {
        "train": name_path("train.tsv.gz", output_dir, prefix),
        "validation": name_path("validation.tsv.gz", output_dir, prefix),
        "test": name_path("test.tsv.gz", output_dir, prefix)
    }


def name_path(suffix, output_dir="./", prefix=None):
    """
    Creates output path using the format /output_dir/prefix.suffix

    Inputs:
        suffix          str: text to append to filename
        output_dir      str: path to output directory
        prefix          str: text to prepend to filename
    Outputs:
        str: formatted filepath 
    """
    return os.path.join(output_dir, ".".join(filter(None, (prefix, suffix))))
                    

def get_file_handle(file_name, mode):
    """
    Returns file handle according to input filetype

    Inputs:
        file_name       str: path to file
        mode            str: which mode to open the file with
    Outputs:

    """
    if file_name.endswith(".gz"):
        return gzip.open(file_name, mode)
    else:
        return open(file_name, mode)


def get_criterion():
    """
    TODO: Move to constants.py?
    """
    return {
        "bcewithlogits": nn.BCEWithLogitsLoss(),
        "crossentropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "pearson": pearson_loss,
        "poissonnll": nn.PoissonNLLLoss()
    }

def get_or_create_dirs(output_path, output_dir):
    """
    """
    new_dir = os.path.join(output_path, output_dir)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    return new_dir


def get_data_splits(data, splits=[80, 10, 10], random_seed=1714):
    """
    """
    # Initialize
    data_splits = [None, None, None]
    train = splits[0] / 100.
    validation = splits[1] / 100.
    test = splits[2] / 100.

    if train == 1.:
        data_splits[0] = data
    elif validation == 1.:
        data_splits[1] = data
    elif test == 1.:
        data_splits[2] = data
    else:
        # Initialize
        p = partial(train_test_split, random_state=random_seed)
        if train == 0.:
            test_size = test
            data_splits[1], data_splits[2] = p(data, test_size=test_size)
        elif validation == 0.:
            test_size = test
            data_splits[0], data_splits[2] = p(data, test_size=test_size)
        elif test == 0.:
            test_size = validation
            data_splits[0], data_splits[1] = p(data, test_size=test_size)
        else:
            test_size = validation + test
            data_splits[0], data = p(data, test_size=test_size)
            test_size = test / (validation + test)
            data_splits[1], data_splits[2] = p(data, test_size=test_size)

    return data_splits


def get_seqs_labels_ids(tsv_file, debugging=False, rev_complement=False,
                        input_length="infer from data"):

    # Sequences / labels / ids
    df = pd.read_table(tsv_file, header=None, comment="#")
    ids = df.pop(0).values
    if input_length != "infer from data":
        seqs = [_resize_sequence(s, input_length) for s in df.pop(1).values]
    else:
        seqs = df.pop(1).values
    seqs = _dna_one_hot_many(seqs)
    labels = df.values

    # Reverse complement
    if rev_complement:
        seqs = np.append(seqs, np.array([s[::-1, ::-1] for s in seqs]), axis=0)
        labels = np.append(labels, labels, axis=0)
        ids = np.append(ids, ids, axis=0)

    # Return 1,000 sequences
    if debugging:
        return seqs[:1000], labels[:1000], ids[:1000]

    return seqs, labels, ids


def _resize_sequence(s, l):

    if len(s) < l:
        return s.center(l, "N")
    elif len(s) > l:
        start = (len(s)//2) - (l//2)
        return s[start:start+l]
    else:
        return s


def _dna_one_hot_many(seqs):
    """One hot encodes a list of sequences."""
    return(np.array([dna_one_hot(str(seq)) for seq in seqs]))



def get_data_loader(seqs, labels, batch_size=100, shuffle=False):

    # TensorDatasets
    dataset = TensorDataset(Tensor(seqs), Tensor(labels))

    # Avoid Error: Expected more than 1 value per channel when training
    batch_size = _avoid_expect_more_than_1_value_per_channel(len(dataset),
        batch_size)
        
    return DataLoader(dataset, batch_size, shuffle=shuffle)


def _avoid_expect_more_than_1_value_per_channel(n, batch_size):

    if n % batch_size == 1:
        return _avoid_expect_more_than_1_value_per_channel(n, batch_size - 1)

    return batch_size


def get_device():

    # Initialize
    device = "cpu"
    free_mems = {}

    # Assign the freest GPU to device
    if cuda.is_available():
        for i in range(cuda.device_count()):
            free_mem = cuda.mem_get_info(f"cuda:{i}")[0]
            free_mems.setdefault(free_mem, [])
            free_mems[free_mem].append(i)
        max_free_mem = max(free_mems.keys())
        random.shuffle(free_mems[max_free_mem])
        device = f"cuda:{free_mems[max_free_mem][0]}"

    return device


def shuffle_string(s, k=2, random_seed=1714):

    # Shuffle
    l = [s[i-k:i] for i in range(k, len(s)+k, k)]
    random.Random(random_seed).shuffle(l)

    return "".join(l)