#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn
from torch.optim.swa_utils import AveragedModel

import modelconfigs
from model_pytorch import Model
from metrics_pytorch import Metrics
import data_processing_pytorch
from load_model import load_model

# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

    description = """
    Test neural net on Go positions from npz files of batches from selfplay.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-npz', help='NPZ file to evaluate', required=True)
    parser.add_argument('-checkpoint', help='Checkpoint to test', required=False)
    parser.add_argument('-pos-len', help='Spatial length of expected training data', type=int, required=True)
    parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
    parser.add_argument('-gpu-idx', help='GPU idx', type=int, required=False)

    args = vars(parser.parse_args())

def main(args):
    npz_file = args["npz"]
    checkpoint_file = args["checkpoint"]
    pos_len = args["pos_len"]
    use_swa = args["use_swa"]
    gpu_idx = args["gpu_idx"]

    world_size = 1
    rank = 0

    # SET UP LOGGING -------------------------------------------------------------

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
        ],
    )
    np.set_printoptions(linewidth=150)

    # FIGURE OUT GPU ------------------------------------------------------------
    if gpu_idx is not None:
        torch.cuda.set_device(gpu_idx)
        logging.info("Using GPU device: " + torch.cuda.get_device_name())
        device = torch.device("cuda", gpu_idx)
    elif torch.cuda.is_available():
        logging.info("Using GPU device: " + torch.cuda.get_device_name())
        device = torch.device("cuda")
    else:
        logging.warning("WARNING: No GPU, using CPU")
        device = torch.device("cpu")

    # LOAD MODEL ---------------------------------------------------------------------

    model, swa_model, _ = load_model(checkpoint_file, use_swa, device=device, pos_len=pos_len, verbose=False)
    model_config = model.config

    batch = np.load(npz_file)

    with torch.no_grad():
        model.eval()
        if swa_model is not None:
            swa_model.eval()

        if swa_model is not None:
            model_outputs = swa_model(torch.tensor(batch["binaryInputNCHW"],device=device),torch.tensor(batch["globalInputNC"],device=device))
        else:
            model_outputs = model(torch.tensor(batch["binaryInputNCHW"],device=device),torch.tensor(batch["globalInputNC"],device=device))

        postprocessed = model.postprocess_output(model_outputs)

        results = tuple(tensor.cpu() for tensor in postprocessed[0])

        (
            policy_logits,      # N, num_policy_outputs, move
            value_logits,       # N, {win,loss,noresult}
            td_value_logits,    # N, {long, mid, short} {win,loss,noresult}
            pred_td_score,      # N, {long, mid, short}
            ownership_pretanh,  # N, 1, y, x
            pred_scoring,       # N, 1, y, x
            futurepos_pretanh,  # N, 2, y, x
            seki_logits,        # N, 4, y, x
            pred_scoremean,     # N
            pred_scorestdev,    # N
            pred_lead,          # N
            pred_variance_time, # N
            pred_shortterm_value_error, # N
            pred_shortterm_score_error, # N
            scorebelief_logits, # N, 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)
        ) = results

        for batch_idx in range(batch["binaryInputNCHW"].shape[0]):
            print(f"Batch idx {batch_idx}")

            policy = torch.softmax(policy_logits[batch_idx],dim=1).numpy()
            print("Policy")
            print(f"Pass {policy[0,pos_len*pos_len]}")
            for y in range(pos_len):
                print(" ".join(["%5.2f%%" % (100.0 * prob) for prob in policy[0,y*pos_len:(y+1)*pos_len]]))
            if model_config["version"] >= 12:
                print("LongOptPolicy")
                print(f"Pass {policy[4,pos_len*pos_len]}")
                for y in range(pos_len):
                    print(" ".join(["%5.2f%%" % (100.0 * prob) for prob in policy[4,y*pos_len:(y+1)*pos_len]]))
                print("ShortOptPolicy")
                print(f"Pass {policy[5,pos_len*pos_len]}")
                for y in range(pos_len):
                    print(" ".join("%5.2f%%" % (100.0 * prob) for prob in policy[5,y*pos_len:(y+1)*pos_len]))
            ownership = torch.tanh(ownership_pretanh[batch_idx,0])
            print("Ownership")
            for y in range(pos_len):
                print(" ".join("%6.2f" % (100.0 * prob) for prob in ownership[y]))

            print("Value " + " ".join("%5.2fc" % (100.0 * prob) for prob in torch.softmax(value_logits[batch_idx],dim=0).numpy()))
            print("ScoreMean " + str(pred_scoremean[batch_idx].item()))
            print("ScoreMeanSq " + str(pred_scorestdev[batch_idx].item() ** 2 + pred_scoremean[batch_idx].item() ** 2))
            print("Lead " + str(pred_lead[batch_idx].item()))
            print("Vartime " + str(pred_variance_time[batch_idx].item()))
            print("STWinLossError " + str(100.0 * math.sqrt(pred_shortterm_value_error[batch_idx].item())) + "c")
            print("STScoreError " + str(math.sqrt(pred_shortterm_score_error[batch_idx].item())))

if __name__ == "__main__":
    main(args)
