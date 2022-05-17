# =============================================================================
# IMPORTS
# =============================================================================
import torch.nn as nn
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# =============================================================================
# FUNCTIONS
# =============================================================================
def get_average_activ(self, inputs, outputs):
    """
    Pytorch Hook that will get the average activation for each layer on the current batch
    which can be usefull to get the average usage of the filter
    :param self: pytorch layer, the layer the hook is attached to
    :param inputs: tensor, current input tensor of the layer
    :param outputs: tensor, current output tensor of the layer
    """

    if self.mode[0] == 'Baseline':
        self.register_buffer('average_activation', outputs.mean(0).mean(1))


# hook to assign the filter values to the average activation across batches
def set_filt_to_aver(self, inputs, outputs):
    """
    Pytorch hook to assign the filter values to the average activation across batches
    :param self: pytorch layer, the layer the hook is attached to
    :param inputs: tensor, current input tensor of the layer
    :param outputs: tensor, current output tensor of the layer
    """

    if self.mode[0] == 'Compare':
        outputs[:, self.mode[1], :] = self.average_activation[self.mode[1]]


def nullify_filter_strict(self, input, output):
    """
    Pytorch Hook that will nullify the output of one of the filter indicated in mode for that layer
    :param self: pytorch layer, the layer the hook is attached to
    :param input: tensor, current input tensor of the layer
    :param output: tensor, current output tensor of the layer
    """

    if self.mode[0] == 'Compare':
        output[:, self.mode[1], :] = 0

def get_explainn_predictions(data_loader, model, device, isSigmoid=False):
    """
    Function to get predictions and true labels for all sequences in the given data loader
    :param data_loader: torch DataLoader, data loader with the sequences of interest
    :param model: ExplaiNN model
    :param device: current available device ('cuda:0' or 'cpu')
    :param isSigmoid: boolean, True if the model output is binary
    :return: tuple of numpy.arrays, model predictions and true labels
    """

    running_outputs = []
    running_labels = []

    with torch.no_grad():
        for seq, lbl in data_loader:
            seq = seq.to(device)
            out = model(seq)
            if isSigmoid:
                sigmoid = nn.Sigmoid()
                out = sigmoid(out.detach().cpu())
            else:
                out = out.detach().cpu()
            running_outputs.extend(out.numpy())
            running_labels.extend(lbl.numpy())

    running_labels = np.array(running_labels)
    running_outputs = np.array(running_outputs)

    return running_outputs, running_labels

#
def get_explainn_unit_outputs(data_loader, model, device):
    """
    Function to get predictions and true labels for all sequences in the given data loader
    :param data_loader: torch DataLoader, data loader with the sequences of interest
    :param model: ExplaiNN model
    :param device: current available device ('cuda:0' or 'cpu')
    :return: numpy.array, outputs of individual units, shape (N, U); N - size of the dataset; U - number of units;
    """

    running_cnn_outputs = []
    for seq, lbl in data_loader:
        x = seq.to(device)
        x = x.repeat(1, model.num_cnns, 1)
        cnn_output = model.linears(x)
        running_cnn_outputs.extend(cnn_output.cpu().detach().numpy())

    running_cnn_outputs = np.array(running_cnn_outputs)
    return running_cnn_outputs


#????? can be applied to SingleLayer??
# def get_namlayer_unit_activations(data_loader, model, device):
#     """
#     Function to scan input sequences by NAMLayer model and compute the individual node outputs
#     (activations)
#     :param data_loader: torch DataLoader, the sequence dataset
#     :param model: NAMLayer model
#     :param device: current available device ('cuda:0' or 'cpu')
#     :return: numpy.array, matrix of activations of shape (N, U); N - size of the dataset; U - number of units
#     """
#
#     running_cnn_outputs = []
#
#     for seq, lbl in data_loader:
#         x = seq.to(device)
#
#         x = x.unsqueeze(-1)
#         cnn_output = model.linear(x)
#
#         running_cnn_outputs.extend(cnn_output.cpu().detach().numpy())
#     running_cnn_outputs = np.array(running_cnn_outputs)
#     return running_cnn_outputs

def get_explainn_unit_activations(data_loader, model, device):
    """
    Function to scan input sequences by ExplaiNN model convolutional filters and compute the convolutional units outputs
    (activations)
    :param data_loader: torch DataLoader, the sequence dataset
    :param model: ExplaiNN model
    :param device: current available device ('cuda:0' or 'cpu')
    :return: numpy.array, matrix of activations of shape (N, U, S); N - size of the dataset; U - number of units;
    S - size of the activation map
    """

    running_activations = []

    with torch.no_grad():
        for seq, lbl in tqdm(data_loader, total=len(data_loader)):
            seq = seq.to(device)

            seq = seq.repeat(1, model.num_cnns, 1)
            act = model.linears[:3](seq)

            running_activations.extend(act.cpu().numpy())

    return np.array(running_activations)


def get_pwms_explainn(activations, sequences, y, filter_size):
    """
    Function to convert filter activation values to PWMs
    :param activations: numpy.array, matrix of activations of shape (N, U, S); N - size of the dataset;
    U - number of units; S - size of the activation map
    :param sequences: torch, array of one-hot encoded sequences
    :param y: torch, array of output labels
    :param filter_size: int, size of the filter
    :return: numpy.array, pwm matrices, shape (U, 4, filter_size), where U - number of units
    """

    # find the threshold value for activation
    activation_threshold = 0.5 * np.amax(activations, axis=(0, 2))

    # Get the number of units/filters
    n_filters = activations.shape[1]

    # pad sequences:
    # npad = ((0, 0), (0, 0), (9, 9))
    # sequences = np.pad(sequences, pad_width=npad, mode='constant', constant_values=0)

    pwm = np.zeros((n_filters, 4, filter_size))
    # pfm = np.zeros((n_filters, 4, filter_size))
    n_samples = activations.shape[0]

    activation_indices = []

    for i in tqdm(range(n_filters)):
        # create list to store filter_size bp sequences that activated filter
        act_seqs_list = []

        for j in range(n_samples):
            # find all indices where filter is activated
            indices = np.where(activations[j, i, :] > activation_threshold[i])

            for start in indices[0]:
                activation_indices.append(start)
                end = start + filter_size
                act_seqs_list.append(sequences[j, :, start:end])

        # convert act_seqs from list to array
        if act_seqs_list:
            act_seqs = np.stack(act_seqs_list)
            pwm_tmp = np.sum(act_seqs, axis=0)
            # pfm_tmp = pwm_tmp
            total = np.sum(pwm_tmp, axis=0)
            pwm_tmp = np.nan_to_num(pwm_tmp / total)

            pwm[i] = pwm_tmp
            # pfm[i] = pfm_tmp

    return pwm


def pwm_to_meme(pwm, output_file_path):
    """
    Function to convert pwm array to meme file
    :param pwm: numpy.array, pwm matrices, shape (U, 4, filter_size), where U - number of units
    :param output_file_path: string, the name of the output meme file
    """

    n_filters = pwm.shape[0]
    filter_size = pwm.shape[2]
    meme_file = open(output_file_path, 'w')
    meme_file.write("MEME version 4 \n")

    print('Saved PWM File as : {}'.format(output_file_path))

    for i in range(0, n_filters):
        if np.sum(pwm[i, :, :]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s \n" % i)
            meme_file.write(
                "letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i, :, :], axis=0)))

        for j in range(0, filter_size):
            if np.sum(pwm[i, :, j]) > 0:
                meme_file.write(str(pwm[i, 0, j]) + "\t" + str(pwm[i, 1, j]) + "\t" + str(pwm[i, 2, j]) + "\t" + str(
                    pwm[i, 3, j]) + "\n")

    meme_file.close()

def get_median_unit_importance(activations, model, unit_outputs, target_labels):
    """
    Function to compute median unit importance (unit_output*class weight) of ExplaiNN units
    :param activations: numpy.array, matrix of activations of shape (N, U, S); N - size of the dataset;
    U - number of units; S - size of the activation map
    :param model: ExplaiNN model
    :param unit_outputs: numpy.array, outputs of individual units, shape (N, U); N - size of the dataset; U - number of units;
    :param target_labels: a list with the names of the output nodes;
    :return: numpy.array, a matrix of size (U, O); U - number of units; O - number of labels/classes;
    """

    activation_threshold = 0.5 * np.amax(activations, axis=(0, 2))

    # sequences (their indeces) that highly activated the filter
    res = {}
    for i in range(100):
        inds = []
        for j in range(activations.shape[0]):
            indices = np.where(activations[j, i, :] > activation_threshold[i])
            if indices[0].shape[0] > 0:
                inds.append(j)
        res[i] = inds

    weights = model.final.weight.detach().cpu().numpy()  # -0.035227 0.480355

    feat_imp_median = np.zeros((100, 50))

    for filt in tqdm(range(100)):
        res_distr = {}
        for cl in range(len(target_labels)):
            f_cell = np.multiply(unit_outputs, weights[cl])
            res_distr[target_labels[cl]] = f_cell[:, filt]
            res_distr[target_labels[cl]] = res_distr[target_labels[cl]][res[filt]]
        res_distr = pd.Series(res_distr)

        res_distr = res_distr.apply(lambda x: np.median(x))
        feat_imp_median[filt, :] = res_distr.values

    return feat_imp_median


def get_specific_unit_importance(activations, model, unit_outputs, filt, target_labels):
    """
    Function to compute unit importance (unit_output*class weight) of a particular ExplaiNN unit (indexed at filt)
    :param activations: numpy.array, matrix of activations of shape (N, U, S); N - size of the dataset;
    U - number of units; S - size of the activation map
    :param model: ExplaiNN model
    :param unit_outputs: numpy.array, outputs of individual units, shape (N, U); N - size of the dataset; U - number of units;
    :param filt: int, index of the unit of interest;
    :param target_labels: a list with the names of the output nodes;
    :return: pandas.Series, contains O keys (number of ExplaiNN outputs, labels), each key contains an array of size X,
    where X is equal to the number of sequences that activated the unit of interest (indexed at filt) more than an
    activation threshold
    """

    activation_threshold = 0.5 * np.amax(activations, axis=(0, 2))

    # sequences (their indeces) that highly activated the filter
    res = {}
    for i in range(100):
        inds = []
        for j in range(activations.shape[0]):
            indices = np.where(activations[j, i, :] > activation_threshold[i])
            if indices[0].shape[0] > 0:
                inds.append(j)
        res[i] = inds

    weights = model.final.weight.detach().cpu().numpy()  # -0.035227 0.480355

    res_distr = {}
    for cl in range(len(target_labels)):
        f_cell = np.multiply(unit_outputs, weights[cl])
        res_distr[target_labels[cl]] = f_cell[:, filt]
        res_distr[target_labels[cl]] = res_distr[target_labels[cl]][res[filt]]

    res_distr = pd.Series(res_distr)

    return res_distr