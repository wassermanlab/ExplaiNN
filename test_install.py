import torch
from torch import nn
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn import metrics
import gzip
import logging

from explainn import utils
from explainn import networks
from explainn import train
from explainn import test
from explainn import interpretation

# Setup logging
logging.basicConfig(
    format="{asctime} - {name} - {levelname} - {message}", 
    style="{",
    datefmt="%Y-%m-%d %H:%M", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """
    """
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    logger.info(f"CUDA available: {cuda_available}" if cuda_available else "No CUDA-enabled GPU is available -- running with CPU")
    #assert cuda_available, "Error with submitting the test script"
    if not cuda_available:
        logger.warning("Training on CPU may cause longer waiting time than expected. Estimate: ~30 minutes")

    
    # Get CUDA device properties
    device = torch.device("cuda:0" if cuda_available else "cpu")
    logger.info(f"Using device: {torch.cuda.get_device_name(device)}" if cuda_available else "Using device: CPU")
    
    logger.info("Begin Test Run:")
    
    # Hyper parameters
    num_epochs = 15
    batch_size = 128
    learning_rate = 0.001

    # Change the working directory to explainn in scratch space.
    #explainn_path = os.path.join(os.environ.get("SCRATCH_PATH"), "ExplaiNN")
    #os.chdir(explainn_path)
    
    h5_path = "./data/test/tf_peaks_TEST_sparse_Remap.h5"
    compressed_file = f"{h5_path}.gz"
    if not os.path.exists(h5_path):
        if os.path.exists(compressed_file):
            logger.info(f"Compressed file {compressed_file} found. Decompressing...")
            with gzip.open(compressed_file, 'rb') as f_in, open(h5_path, 'wb') as f_out:
                f_out.write(f_in.read())
            logger.info(f"Decompression complete: {h5_path}")
        else:
            raise FileNotFoundError(f"Neither {h5_path} nor {compressed_file} was found.")

    # Load data
    dataloaders, target_labels, train_out = utils.tools.load_datas(h5_path,
                                                        batch_size,
                                                        0,
                                                        True)
    target_labels = [i.decode("utf-8") for i in target_labels]

    # Model parameters
    num_cnns = 100
    input_length = 200
    num_classes = len(target_labels)
    filter_size = 19
    
    # Create model
    model = networks.ExplaiNN(num_cnns, input_length, num_classes, filter_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    weights_folder = "./data/test/weights"
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    # Train model
    model, train_error, test_error = train.train_explainn(dataloaders["train"],
                                                        dataloaders["valid"],
                                                        model,
                                                        device,
                                                        criterion,
                                                        optimizer,
                                                        num_epochs,
                                                        weights_folder,
                                                        name_ind="",
                                                        verbose=True,
                                                        trim_weights=False,
                                                        checkpoint=0,
                                                        patience=0)

    # Plot loss
    utils.tools.showPlot(train_error, test_error, "Loss trend", "Loss")

    # Test model
    model.load_state_dict(torch.load(f"{weights_folder}/{os.listdir(weights_folder)[0]}"))
    labels_E, outputs_E = test.run_test(model, dataloaders["test"], device)

    # Get metrics
    pr_rec = average_precision_score(labels_E, outputs_E)
    no_skill_probs = [0 for _ in range(len(labels_E[:, 0]))]
    ns_fpr, ns_tpr, _ = metrics.roc_curve(labels_E[:, 0], no_skill_probs)

    roc_aucs, raw_aucs, roc_prcs, raw_prcs = {}, {}, {}, {}
    for i in range(len(target_labels)):
        nn_fpr, nn_tpr, threshold = metrics.roc_curve(labels_E[:, i], outputs_E[:, i])
        roc_auc_nn = metrics.auc(nn_fpr, nn_tpr)

        precision_nn, recall_nn, thresholds = metrics.precision_recall_curve(labels_E[:, i], outputs_E[:, i])
        pr_auc_nn = metrics.auc(recall_nn, precision_nn)

        raw_aucs[target_labels[i]] = nn_fpr, nn_tpr
        roc_aucs[target_labels[i]] = roc_auc_nn

        raw_prcs[target_labels[i]] = recall_nn, precision_nn
        roc_prcs[target_labels[i]] = pr_auc_nn

    logger.info(roc_prcs)
    logger.info(roc_aucs)
    
    logger.info("Testing Complete")


if __name__=='__main__':
    main()