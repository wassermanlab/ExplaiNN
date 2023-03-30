# =============================================================================
# IMPORTS
# =============================================================================
import torch
import numpy as np
import h5py
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# =============================================================================
# FUNCTIONS
# =============================================================================
def count_parameters(model):
    """
    Calculates the number of parameters in the model

    :param model: pytorch model
    :return: int, number of parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def showPlot(points, points2, title, save=None):
    """
    function to show training curve
    save - place to save the figure
    :param points: list, training loss
    :param points2: list, validation loss
    :param title: string, optional title
    :param save: boolean, save figure as a sep file or not
    :return:
    """

    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.plot(points2)
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.title(title)

    if save:
        plt.savefig(save)
    else:
        plt.show()


def load_datas(path_h5, batch_size, num_workers, shuffle):
    """
    Loads the dataset from an h5 file

    :param path_h5: string, path to the file
    :param batch_size: int, batch size to use
    :param num_workers: int, number of workers for DataLoader
    :param shuffle: boolean, shuffle parameter in DataLoader
    :return: tuple:
        a dictionary with train, validation, and test sets;
        a list with the names of the output nodes;
        an array with true values for the train samples;
    """

    data = h5py.File(path_h5, 'r')
    dataset = {}
    dataloaders = {}
    # Train data
    dataset['train'] = torch.utils.data.TensorDataset(torch.Tensor(np.array(data['train_in'])),
                                                      torch.Tensor(np.array(data['train_out'])))
    dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                       batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)

    # Validation data
    dataset['valid'] = torch.utils.data.TensorDataset(torch.Tensor(np.array(data['valid_in'])),
                                                      torch.Tensor(np.array(data['valid_out'])))
    dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
                                                       batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)

    # Test data
    dataset['test'] = torch.utils.data.TensorDataset(torch.Tensor(np.array(data['test_in'])),
                                                     torch.Tensor(np.array(data['test_out'])))
    dataloaders['test'] = torch.utils.data.DataLoader(dataset['test'],
                                                      batch_size=batch_size, shuffle=shuffle,
                                                      num_workers=num_workers)
    print('Dataset Loaded')
    target_labels = list(data['target_labels'])
    train_out = np.array(data['train_out'])
    return dataloaders, target_labels, train_out


def load_single_data(path_h5, batch_size, num_workers, shuffle):
    """
    Load h5 file as a single dataset. Has to have the following fields:
        train_in : input train data
        valid_in : input validation data
        test_in : input test data
        train_out : output train labels
        valid_out : output validation labels
        test_out : output test labels
    :param path_h5: string, path to h5 file with the data
    :param batch_size: int, size of the batch in the dataloader
    :param num_workers: int, number of workers for DataLoader
    :param shuffle: boolean, shuffle parameter in DataLoader
    :return: tuple:
                DataLoader, a dataset where all train, validation, and test samples are put together
                torch.tensor, all input data
                torch.tensor, all labels of input data
    """

    data = h5py.File(path_h5, 'r')

    x = torch.Tensor(np.array(data['train_in']))
    y = torch.Tensor(np.array(data['valid_in']))
    z = torch.Tensor(np.array(data['test_in']))

    x_lab = torch.Tensor(np.array(data['train_out']))
    y_lab = torch.Tensor(np.array(data['valid_out']))
    z_lab = torch.Tensor(np.array(data['test_out']))

    res = torch.cat((x, y, z), dim=0)
    res_lab = torch.cat((x_lab, y_lab, z_lab), dim=0)

    all_dataset = torch.utils.data.TensorDataset(res, res_lab)
    dataloader  = torch.utils.data.DataLoader(all_dataset,
                                                  batch_size=128, shuffle=False,
                                                  num_workers=0)

    return dataloader, res, res_lab


def dna_one_hot(seq, seq_len=None, flatten=False):
    """
    Converts an input dna sequence to one hot encoded representation, with (A:0,C:1,G:2,T:3) alphabet

    :param seq: string, input dna sequence
    :param seq_len: int, optional, length of the string
    :param flatten: boolean, if true, makes a 1 column vector
    :return: numpy.array, one-hot encoded matrix of size (4, L), where L - the length of the input sequence
    """

    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim:seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    seq = seq.replace("A", "0")
    seq = seq.replace("C", "1")
    seq = seq.replace("G", "2")
    seq = seq.replace("T", "3")

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    # dtype="int8" fails for N's
    seq_code = np.zeros((4, seq_len), dtype="float16")
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:, i] = 0.
        else:
            try:
                seq_code[int(seq[i - seq_start]), i] = 1.
            except:
                seq_code[:, i] = 0.

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_code = seq_code.flatten()[None, :]

    return seq_code


def convert_one_hot_back_to_seq(dataloader):
    """
    Converts one-hot encoded matrices back to DNA sequences
    :param dataloader: pytorch, DataLoader
    :return: list of strings, DNA sequences
    """

    sequences = []
    code = list("ACGT")
    for seqs, labels in tqdm(dataloader, total=len(dataloader)):
        x = seqs.permute(0, 1, 3, 2)
        x = x.squeeze(-1)
        for i in range(x.shape[0]):
            seq = ""
            for j in range(x.shape[-1]):
                try:
                    seq = seq + code[int(np.where(x[i, :, j] == 1)[0])]
                except:
                    print("error")
                    print(x[i, :, j])
                    print(np.where(x[i, :, j] == 1))
                    break
            sequences.append(seq)
    return sequences


def _flip(x, dim):
    """
    Adapted from Selene:
    https://github.com/FunctionLab/selene/blob/master/selene_sdk/utils/non_strand_specific_module.py

    Reverses the elements in a given dimension `dim` of the Tensor.
    source: https://github.com/pytorch/pytorch/issues/229
    """

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
        x.size(0), x.size(1), -1)[:, getattr(
        torch.arange(x.size(1) - 1, -1, -1),
        ("cpu", "cuda")[x.is_cuda])().long(), :]

    return x.view(xsize)


def pearson_loss(x, y):
    """
    Loss that is based on Pearson correlation/objective function
    :param x: torch, input data
    :param y: torch, output labels
    :return: torch, pearson loss per sample
    """

    mx = torch.mean(x, dim=1, keepdim=True)
    my = torch.mean(y, dim=1, keepdim=True)
    xm, ym = x - mx, y - my

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = torch.sum(1-cos(xm,ym))
    return loss


def change_grad_filters(model, val, index):
    """
    Function to modify the gradient of filters of interest by the specified value
    :param model: ExplaiNN model
    :param val: float or int, value by which gradients will be multiplied
    :param index: list, gradients of which unit filters to modify
    :return: ExplaiNN model
    """

    def replace_val(grad, val, index):
        grad[index, :, :] *= val

        return grad

    model.linears[0].weight.register_hook(lambda grad: replace_val(grad, val, index))

    return model


def _PWM_to_filter_weights(pwm, filter_size=19):
    """
    Function to convert a given pwm into convolutional filter
    :param pwm: list of length L (size of the PWM) with lists of size 4 (nucleotide values)
    :param filter_size: int, the size of the pwm
    :return: numpy.array, of shape (L, 4)
    """

    # Initialize
    lpop = 0
    rpop = 0

    pwm = [[.25,.25,.25,.25]]*filter_size+pwm+[[.25,.25,.25,.25]]*filter_size

    while len(pwm) > filter_size:
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

    return(np.array(pwm) - .25)


def read_meme(meme_file):
    """
    Function the motifs from the input meme file
    Right now works only if motifs are of the same length
    :param meme_file: string, path to the meme file
    :return: tuple:
                numpy.array, matrix of pwms of size (N, L, 4), where N is the number of motifs, L is the motif length,
                4 number of nucleotides
    """

    with open(meme_file) as fp:
        line = fp.readline()
        motifs = []
        motif_names = []
        while line:
            # determine length of next motif
            if line.split(" ")[0] == 'MOTIF':
                line = line.strip()
                # add motif number to separate array
                motif_names.append(line.split(" ")[1])
                # get length of motif
                line2 = fp.readline().split(" ")
                motif_length = int(float(line2[5]))
                # read in motif
                current_motif = np.zeros((motif_length, 4))  # Edited pad shorter ones with 0
                for i in range(motif_length):
                    current_motif[i, :] = fp.readline().strip().split()
                motifs.append(current_motif)
            line = fp.readline()
        motifs = np.stack(motifs)
        motif_names = np.stack(motif_names)

    return motifs, motif_names
