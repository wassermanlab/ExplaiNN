#!/usr/bin/env python
import os
import click
import json
import logging

from explainn.train.train import train_explainn
from explainn.utils.tools import pearson_loss
from explainn.models.networks import ExplaiNN
from explainn.parsers.preprocess import combine_seq_files
from explainn.parsers.parse import json2explainn

from train import run_train
from test import test_model
from interpret import interpret_results
from utils import save_data_splits, validate_config


# Setup logging
logging.basicConfig(
    format="{asctime} - {name} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "config_file",
    type=click.Path(exists=True, resolve_path=True),
)
def main(**args):
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

    # TODO: Add preprocessing steps as arguments/config, eg. match-seqs-by-gc,
    # subsample-seqs-by-gc, resize, etc.
    if config["preprocessing"]["match_seqs_by_gc"]:
        # TODO: perform match seqs by gc
        pass
    if config["preprocessing"]["subsample_seqs_by_gc"]:
        # TODO: perform subsample_seqs_by_gc
        pass
    if config["preprocessing"]["resize"]:
        # TODO Perform resize? 
        pass
    
    # Preprocess the data
    # TODO: Add this as an argument/in config
    classes = combine_seq_files(config["data"]["input_files"])
    splits = json2explainn(classes)
    save_data_splits(
        config["data"]["output_dir"],
        splits[0],
        splits[1],
        splits[2],
        config["data"]["prefix"],
    )
    # TODO: Update config file with output location? Where to store path to intermediates

    if config["options"]["store_intermediates"]:
        handle = open(
            os.path.join(config["data"]["output_dir"], "combined_data.json"), "wt"
        )
        json.dump(classes, handle, indent=4, sort_keys=True)
        handle.close()

    # Train the model
    run_train(config)

    # Test the model
    test_model(config)

    # Interpret the results
    interpret_results(config)

    # Finetune the model
    # TODO: Specify this with config/arguments

    # Further interpretation
    # TODO: Specify these with config/arguments
    # MEME to logos
    meme2logo(config)

    # MEME to scores
    # meme2scores(config)

    # MEME to clusters
    # meme2clusters(config)

    # Tomtom
    # tomtom(config)

    # JASPAR to logos
    # jaspar2logo(config)

    # PWM to scores
    # pwm2scores(config)


if __name__ == "__main__":
    main()
