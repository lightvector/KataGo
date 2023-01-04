import json
import logging
import os

import torch
from torch.optim.swa_utils import AveragedModel

import modelconfigs
from model_pytorch import Model, ResBlock, NestedBottleneckResBlock

def load_model(checkpoint_file, use_swa, device, pos_len=19, verbose=False):
  state_dict = torch.load(checkpoint_file,map_location="cpu")

  if "config" in state_dict:
    model_config = state_dict["config"]
  else:
    config_file = os.path.join(os.path.dirname(checkpoint_file),"model.config.json")
    logging.info(f"No config in checkpoint, so loading from: {config_file}")
    with open(config_file,"r") as f:
      model_config = json.load(f)

  logging.info(str(model_config))
  model = Model(model_config,pos_len)
  model.initialize()

  # Strip off any "module." from when the model was saved with DDP or other things
  model_state_dict = {}
  for key in state_dict["model"]:
    old_key = key
    while key.startswith("module."):
      key = key[7:]
    model_state_dict[key] = state_dict["model"][old_key]
  model.load_state_dict(model_state_dict)

  model.to(device)

  swa_model = None
  if use_swa:
    if state_dict is None:
      raise Exception("Cannot use swa without a trained model")
    if "swa_model" not in state_dict:
      raise Exception("Checkpoint doesn't contain swa_model")
    swa_model = AveragedModel(model, device=device)
    swa_model.load_state_dict(state_dict["swa_model"])

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
  other_state_dict["metrics"] = state_dict["metrics"]
  other_state_dict["running_metrics"] = state_dict["running_metrics"]
  other_state_dict["train_state"] = state_dict["train_state"]

  return (model, swa_model, other_state_dict)
