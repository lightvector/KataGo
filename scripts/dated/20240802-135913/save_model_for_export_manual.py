#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import contextlib
import json
import datetime
from datetime import timezone
import gc
import shutil
import glob
import numpy as np
import itertools
import copy
import atexit
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing

import modelconfigs
from model_pytorch import Model
from metrics_pytorch import Metrics
import load_model
import data_processing_pytorch

# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

    description = """
    Script for one-off taking a checkpoint training under train.py and making it save itself for one export,
    without having to use train.py arguments where it would do so repeatedly periodically.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
    parser.add_argument('-exportdir', help='Directory to export models periodically', required=True)
    parser.add_argument('-exportprefix', help='Prefix to append to names of models', required=True)

    args = vars(parser.parse_args())


def make_dirs(args):
    exportdir = args["exportdir"]

    if exportdir is not None and not os.path.exists(exportdir):
        os.makedirs(exportdir)


def main(args):
    traindir = args["traindir"]
    exportdir = args["exportdir"]
    exportprefix = args["exportprefix"]

    # SET UP LOGGING -------------------------------------------------------------

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(traindir,f"save_model_for_export_manual.log"), mode="a"),
            logging.StreamHandler()
        ],
    )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # LOAD MODEL ---------------------------------------------------------------------

    def get_checkpoint_path():
        return os.path.join(traindir,"checkpoint.ckpt")

    def save(model_state_dict, swa_model_state_dict, optimizer_state_dict, metrics_obj_state_dict, running_metrics, train_state, last_val_metrics, path):
        assert path is not None

        state_dict = {}
        state_dict["model"] = model_state_dict
        state_dict["optimizer"] = optimizer_state_dict
        state_dict["metrics"] = metrics_obj_state_dict
        state_dict["running_metrics"] = running_metrics
        state_dict["train_state"] = train_state
        state_dict["last_val_metrics"] = last_val_metrics
        state_dict["config"] = model_config

        if swa_model_state_dict is not None:
            state_dict["swa_model"] = swa_model_state_dict

        logging.info("Saving checkpoint: " + path)
        torch.save(state_dict, path + ".tmp")
        time.sleep(1)
        os.replace(path + ".tmp", path)

    def load():
        path_to_load_from = get_checkpoint_path()
        assert path_to_load_from is not None

        state_dict = torch.load(path_to_load_from, map_location="cpu")
        model_config = state_dict["config"]
        logging.info(str(model_config))

        train_state = state_dict["train_state"]
        model_state_dict = state_dict["model"]
        swa_model_state_dict = None
        if "swa_model" in state_dict:
            swa_model_state_dict = state_dict["swa_model"]

        metrics_obj_state_dict = state_dict["metrics"]
        running_metrics = state_dict["running_metrics"]

        optimizer_state_dict = state_dict["optimizer"]
        last_val_metrics = state_dict["last_val_metrics"]

        return (model_config, model_state_dict, swa_model_state_dict, optimizer_state_dict, metrics_obj_state_dict, running_metrics, train_state, last_val_metrics)

    (model_config, model_state_dict, swa_model_state_dict, optimizer_state_dict, metrics_obj_state_dict, running_metrics, train_state, last_val_metrics) = load()

    assert "global_step_samples" in train_state
    assert "total_num_data_rows" in train_state

    logging.info("=========================================================================")
    logging.info("SAVING MODEL FOR EXPORT MANUAL")
    logging.info("=========================================================================")
    logging.info("Current time: " + str(datetime.datetime.now()))
    logging.info("Global step: %d samples" % (train_state["global_step_samples"]))
    logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
    logging.info(f"Training dir: {traindir}")
    logging.info(f"Export dir: {exportdir}")

    # Export a model for testing, unless somehow it already exists
    modelname = "%s-s%d-d%d" % (
        exportprefix,
        train_state["global_step_samples"],
        train_state["total_num_data_rows"],
    )
    savepath = os.path.join(exportdir,modelname)
    savepathtmp = os.path.join(exportdir,modelname+".tmp")
    if os.path.exists(savepath):
        logging.info("NOT saving model, already exists at: " + savepath)
    else:
        os.mkdir(savepathtmp)
        logging.info("SAVING MODEL FOR EXPORT TO: " + savepath)
        save(model_state_dict, swa_model_state_dict, optimizer_state_dict, metrics_obj_state_dict, running_metrics, train_state, last_val_metrics, path=os.path.join(savepathtmp,"model.ckpt"))
        time.sleep(2)
        os.rename(savepathtmp,savepath)



if __name__ == "__main__":
    make_dirs(args)
    main(args)
