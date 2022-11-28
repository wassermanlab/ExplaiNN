import click
import gzip
from functools import partial
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from torch import cuda, Tensor
from torch.utils.data import DataLoader, TensorDataset

from explainn.utils.tools import dna_one_hot

###############################################################################
def click_validator(validator):

    class Validator(click.Command):

        def make_context(self, *args, **kwargs):
            context = super(Validator, self).make_context(*args, **kwargs)
            validator(context)

            return(context)

    return Validator


###############################################################################
def get_file_handle(file_name, mode):

    if file_name.endswith(".gz"):
        handle = gzip.open(file_name, mode)
    else:
        handle = open(file_name, mode)

    return handle


###############################################################################
def get_data_splits(data, splits=[80, 10, 10], random_seed=1714):

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


###############################################################################
def get_seqs_labels_ids(tsv_file, debugging=False, rev_complement=False,
                        input_length="infer from data"):

    # Sequences / labels / ids
    df = pd.read_table(tsv_file, header=None)
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


###############################################################################
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


###############################################################################
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

###############################################################################
def shuffle_string(s, k=2, random_seed=1714):

    # Shuffle
    l = [s[i-k:i] for i in range(k, len(s)+k, k)]
    random.Random(random_seed).shuffle(l)

    return "".join(l)