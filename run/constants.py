import torch
from torch import nn
from explainn.utils.tools import pearson_loss

CRITERIONS = {
    "bcewithlogits": nn.BCEWithLogitsLoss(),
    "crossentropy": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "pearson": pearson_loss,
    "poissonnll": nn.PoissonNLLLoss(),
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

CONFIG_REQUIRED_FIELDS = {
    "data": {
        "input_files": list,
        "output_dir": str,
        "prefix": str,
        "rev_complement": bool,
        "input_length": int,
        "intermediates": {
            "training_file": str,
            "validation_file": str,
            "test_file": str,
        },
    },
    "cnn": {
        "filter_size": int,
        "num_fc": int,
        "num_units": int,
        "pool_size": int,
        "pool_stride": int,
    },
    "training": {
        "cpu_threads": int,
        "batch_size": int,
        "num_epochs": int,
        "checkpoint": int,
        "patience": int,
        "trim_weights": bool,
    },
    "optimizer": {"criterion": str, "lr": float, "optimizer": str},
    "interpretation": {
        "model_file": str,
        "cpu_threads": int,
        "batch_size": int,
        "num_well_pred_seqs": int,
        "correlation": int,
        "exact_match": bool,
        "percentile_bottom": int,
        "percentile_top": int,
    },
    "options": {"debugging": bool, "use_time": bool, "store_intermediates": bool},
    "postprocess": {
        "cpu_threads": int,
        "target_file": str,
        "tomtom": {
            "dist": str,
            "evalue": bool,
            "min_overlap": int,
            "motif_pseudo": float,
            "threshold": float,
        },
    },
}
