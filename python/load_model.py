import json
import logging
import os

import torch
from torch.optim.swa_utils import AveragedModel

import modelconfigs
from model_pytorch import Model, ResBlock, NestedBottleneckResBlock

def load_model_state_dict(state_dict):
    # Strip off any "module." from when the model was saved with DDP or other things
    model_state_dict = {}
    for key in state_dict["model"]:
        old_key = key
        while key.startswith("module."):
            key = key[7:]
        # Filter out some extra keys that were present in older checkpoints
        if "score_belief_offset_vector" in key or "score_belief_offset_bias_vector" in key or "score_belief_parity_vector" in key:
            continue
        model_state_dict[key] = state_dict["model"][old_key]
    return model_state_dict

def load_swa_model_state_dict(state_dict):
    if "swa_model" not in state_dict:
        return None
    swa_model_state_dict = {}
    for key in state_dict["swa_model"]:
        # Filter out some extra keys that were present in older checkpoints
        if "score_belief_offset_vector" in key or "score_belief_offset_bias_vector" in key or "score_belief_parity_vector" in key:
            continue
        swa_model_state_dict[key] = state_dict["swa_model"][key]
    return swa_model_state_dict


def load_model(checkpoint_file, use_swa, device, pos_len=19, for_coreml=False, verbose=False):
    state_dict = torch.load(checkpoint_file,map_location="cpu")

    if "config" in state_dict:
        model_config = state_dict["config"]
    else:
        config_file = os.path.join(os.path.dirname(checkpoint_file),"model.config.json")
        logging.info(f"No config in checkpoint, so loading from: {config_file}")
        with open(config_file,"r") as f:
            model_config = json.load(f)

    logging.info(str(model_config))
    model = Model(model_config,pos_len,for_coreml=for_coreml)
    model.initialize()

    # Strip off any "module." from when the model was saved with DDP or other things
    model_state_dict = load_model_state_dict(state_dict)
    model.load_state_dict(model_state_dict)

    model.to(device)

    swa_model = None
    if use_swa:
        if state_dict is None:
            raise Exception("Cannot use swa without a trained model")
        swa_model_state_dict = load_swa_model_state_dict(state_dict)
        if swa_model_state_dict is None:
            raise Exception("Checkpoint doesn't contain swa_model")
        swa_model = AveragedModel(model, device=device)
        swa_model.load_state_dict(swa_model_state_dict)

        # Grab the model out of the SWA so that we have all the methods of the original model.
        swa_model = swa_model.module

    if verbose:
        total_num_params = 0
        total_trainable_params = 0
        logging.info("Parameters in model:")
        for name, param in model.named_parameters():
            product = 1
            for dim in param.shape:
                product *= int(dim)
            if param.requires_grad:
                total_trainable_params += product
            total_num_params += product
            logging.info(f"{name}, {list(param.shape)}, {product} params")
        logging.info(f"Total num params: {total_num_params}")
        logging.info(f"Total trainable params: {total_trainable_params}")

    # Return other useful stuff in state dict too
    other_state_dict = {}
    if "metrics" in state_dict:
        other_state_dict["metrics"] = state_dict["metrics"]
    if "running_metrics" in state_dict:
        other_state_dict["running_metrics"] = state_dict["running_metrics"]
    if "train_state" in state_dict:
        other_state_dict["train_state"] = state_dict["train_state"]
    if "last_val_metrics" in state_dict:
        other_state_dict["last_val_metrics"] = state_dict["last_val_metrics"]

    return (model, swa_model, other_state_dict)
