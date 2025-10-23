#!/usr/bin/python3
# Example usage:
# wget https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b40c256-s11840935168-d2898845681.zip
# unzip kata1-b40c256-s11840935168-d2898845681.zip
# python python/convert_coreml.py -saved-model-dir kata1-b40c256-s11840935168-d2898845681/saved_model -name-scope swa_model -board_size 19

import argparse
import json
import tensorflow as tf

from model import Model

import common
import tempfile
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
import coremltools as ct

description = """
Convert a trained neural net to a CoreML model.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
parser.add_argument('-board-size', help='Board size of model', required=False)
args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]
pos_len = int(args["board_size"])

if pos_len is None:
  pos_len = 19

# Model ----------------------------------------------------------------

with open(model_config_json) as f:
  model_config = json.load(f)

if name_scope is not None:
  with tf.compat.v1.variable_scope(name_scope):
    model = Model(model_config,pos_len,{})
else:
  model = Model(model_config,pos_len,{})

saver = tf.compat.v1.train.Saver(
  max_to_keep = 10000,
  save_relative_paths = True,
)

model_dir = tempfile.mkdtemp()
graph_def_file = os.path.join(model_dir, 'tf_graph.pb')
checkpoint_file = os.path.join(model_dir, 'tf_model.ckpt')
frozen_graph_file = os.path.join(model_dir, 'KataGoModel.pb')
mlmodel_file = f'KataGoModel{pos_len}x{pos_len}.mlmodel'

output_names = [
  model.policy_output.op.name,
  model.value_output.op.name,
  model.ownership_output.op.name,
  model.miscvalues_output.op.name,
  model.moremiscvalues_output.op.name
]

print(output_names)
with tf.compat.v1.Session() as session:
  saver.restore(session, model_variables_prefix)

  tf.train.write_graph(session.graph, model_dir, graph_def_file, as_text=False)
  # save the weights
  saver = tf.train.Saver()
  saver.save(session, checkpoint_file)

  # take the graph definition and weights
  # and freeze into a single .pb frozen graph file
  freeze_graph(input_graph=graph_def_file,
               input_saver="",
               input_binary=True,
               input_checkpoint=checkpoint_file,
               output_node_names=','.join(output_names),
               restore_op_name="save/restore_all",
               filename_tensor_name="save/Const:0",
               output_graph=frozen_graph_file,
               clear_devices=True,
               initializer_nodes="")

  mlmodel = ct.convert(frozen_graph_file)
  mlmodel.short_description = f'KataGo {pos_len}x{pos_len} model version {model.version} converted from {model_config_json}'
  mlmodel.version = f'{model.version}'
  mlmodel.save(mlmodel_file)

  print("Core ML model saved at {}".format(mlmodel_file))
