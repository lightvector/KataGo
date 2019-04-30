import sys
import os
import argparse
import traceback
import json

def add_model_load_args(argparser):
  argparser.add_argument('-saved-model-dir', help='Parent dir containing model.config.json and variables/variables.*"', required=False)
  argparser.add_argument('-model-variables-prefix', help='Direct path to model variables files, excluding the ".index" or ".data-00000-of-00001"', required=False)
  argparser.add_argument('-model-config-json', help='Direct path to model.config.json to use', required=False)

def load_model_paths(args):
  saved_model_dir = args["saved_model_dir"]
  model_variables_prefix = args["model_variables_prefix"]
  model_config_json = args["model_config_json"]

  saved_model_dir_specified = saved_model_dir is not None
  direct_paths_specified = model_variables_prefix is not None and model_config_json is not None
  if saved_model_dir_specified == direct_paths_specified:
    raise Exception("Must specify exactly one of -saved-model-dir OR -model-variables-prefix AND -model-config-json")

  if saved_model_dir is not None:
    return (os.path.join(saved_model_dir,"variables","variables"), os.path.join(saved_model_dir,"model.config.json"))
  else:
    return (model_variables_prefix, model_config_json)

