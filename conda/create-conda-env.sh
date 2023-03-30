#!/usr/bin/env bash

# Create conda environment
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
conda env create -f ${SCRIPT_DIR}/${HOSTNAME}.yml
