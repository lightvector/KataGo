#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import tensorflow as tf
import numpy as np

from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves

#Command and args-------------------------------------------------------------------

description = """
Train neural net on Go games!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-weightsdir', help='Dir to write for training weights', required=True)
parser.add_argument('-gamesdir', help='Dir of games to read', required=True, action='append')
parser.add_argument('-verbose', help='verbose', required=False, action='store_true')
args = vars(parser.parse_args())

weightsdir = args["weightsdir"]
gamesdirs = args["gamesdir"]
verbose = args["verbose"]

#Data loading-------------------------------------------------------------------

def load_sgf_moves(path):
  sgf_file = open(path,"rb")
  contents = sgf_file.read()
  sgf_file.close()
  try:
    game = sgf.Sgf_game.from_bytes(contents)
  except:
    if verbose:
      traceback.print_exc()
    raise Exception("Error parsing sgf file: " + path)
  return sgf_moves.get_setup_and_moves(game)

def collect_game_files(gamesdir):
  files = []
  for root, directories, filenames in os.walk(gamesdir):
    for filename in filenames:
      files.append(os.path.join(root,filename))
  return files

game_files = []
for gamesdir in gamesdirs:
  print("Collecting games in " + gamesdir, flush=True)
  files = collect_game_files(gamesdir)
  files = [path for path in files if path.endswith(".sgf")]
  game_files.extend(files)
  print("Collected %d games" % (len(files)), flush=True)

print("Total: collected %d games" % (len(game_files)), flush=True)


#Feature extraction functions-------------------------------------------------------------------

#Neural net inputs
#19x19 is on board
#19x19 own stone present
#19x19 opp stone present

#Maybe??
#19x19x5 own stone present 0-4 turns ago
#19x19x5 opp stone present 0-4 turns ago
#19x19xn one-hot encoding of various ranks
#19x19xn some encoding of komi
#19x19x4 own ladder going through this spot in each direction would work (nn,np,pn,pp)
#19x19x4 opp ladder going through this spot in each direction would work (nn,np,pn,pp)

#Neural net outputs
#19x19 move
#1 pass #TODO

#TODO data symmetrizing
#TODO use more data
#TODO more data features?? definitely at least history
#TODO test different neural net structures, particularly the final combining layer
#TODO weight and neuron activation visualization
#TODO keep tabs on dead neurons and such
#TODO save weights and such, keep a notes file of configurations and results
#TODO run same NN several times to get an idea of consistency
#TODO validation set in addition to training set
#TODO batch normalization
#TODO try residual structure?
#TODO gpu-acceleration!

max_size = 19
input_shape = [19,19,3]
target_shape = [19*19]

#TODO don't assume 19
def fill_row_features(board, pla, opp, next_loc, input_data, target_data, target_data_weights, idx):
  for y in range(19):
    for x in range(19):
      input_data[idx,y,x,0] = 1.0
      color = board.get(y,x)
      if color == pla:
        input_data[idx,y,x,1] = 1.0
      elif color == opp:
        input_data[idx,y,x,2] = 1.0

  if next_loc is None:
    # TODO for now we weight these rows to 0
    target_data[idx,0] = 1.0
    target_data_weights[idx] = 0.0
    pass
    # target_data[idx,max_size*max_size] = 1.0
  else:
    (y,x) = next_loc
    target_data[idx,y*max_size+x] = 1.0
    target_data_weights[idx] = 1.0

def fill_features(num_rows, prob_to_include_row, input_data, target_data, target_data_weights):
  idx = 0
  for filename in game_files:

    (board,moves) = load_sgf_moves(filename)
    for (color,next_loc) in moves:

      if random.random() < prob_to_include_row:
        if color == 'b':
          pla = 'b'
          opp = 'w'
        elif color == 'w':
          opp = 'b'
          pla = 'w'
        else:
          assert False

        fill_row_features(board,pla,opp,next_loc,input_data,target_data,target_data_weights,idx)
        idx += 1
        if idx >= num_rows:
          return

      if next_loc is not None: # pass
        (row,col) = next_loc
        board.play(row,col,color)

  assert(idx == num_rows)


# Build model -------------------------------------------------------------

print("Building model", flush=True)

def init_stdev(num_inputs,num_outputs):
  #xavier
  #return math.sqrt(2.0 / (num_inputs + num_outputs))
  #herangzhen
  return math.sqrt(2.0 / (num_inputs))

def weight_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs)
  initial = tf.truncated_normal(shape=shape, stddev=stdev)
  return tf.Variable(initial,name=name)

def bias_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 2.0
  initial = tf.truncated_normal(shape=shape, mean=stdev, stddev=stdev)
  return tf.Variable(initial,name=name)

def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

#Input layer
inputs = tf.placeholder(tf.float32, [None] + input_shape)

cur_layer = inputs
cur_num_channels = input_shape[2]

#Convolutional RELU layer 1
conv1diam = 3
conv1num_channels = 16
conv1w = weight_variable("conv1w",[conv1diam,conv1diam,cur_num_channels,conv1num_channels],cur_num_channels,conv1num_channels)
conv1b = bias_variable("conv1b",[conv1num_channels],cur_num_channels,conv1num_channels)

cur_layer = tf.nn.relu(conv2d(cur_layer, conv1w) + conv1b)
cur_num_channels = conv1num_channels

#Convolutional RELU layer 2
conv2diam = 3
conv2num_channels = 8
conv2w = weight_variable("conv2w",[conv2diam,conv2diam,cur_num_channels,conv2num_channels],cur_num_channels,conv2num_channels)
conv2b = bias_variable("conv2b",[conv2num_channels],cur_num_channels,conv2num_channels)

cur_layer = tf.nn.relu(conv2d(cur_layer, conv2w) + conv2b)
cur_num_channels = conv2num_channels

#Convolutional linear layer 3
conv3diam = 5
conv3num_channels = 1
conv3w = weight_variable("conv3w",[conv3diam,conv3diam,cur_num_channels,conv3num_channels],cur_num_channels,conv3num_channels)
conv3b = bias_variable("conv3b",[conv3num_channels],cur_num_channels,conv3num_channels)

cur_layer = conv2d(cur_layer, conv3w) + conv3b
cur_num_channels = conv3num_channels

#Output
assert(cur_num_channels == 1)
output_layer = tf.reshape(cur_layer, [-1] + target_shape)

#Loss function
targets = tf.placeholder(tf.float32, [None] + target_shape)
target_weights = tf.placeholder(tf.float32, [None])
loss = tf.reduce_mean(target_weights * tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output_layer))

#Results
batch_learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(batch_learning_rate).minimize(loss)
target_idxs = tf.argmax(targets, 1)
top1_prediction = tf.equal(tf.argmax(output_layer, 1), target_idxs)
top4_prediction = tf.nn.in_top_k(output_layer,target_idxs,4)
accuracy1 = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
accuracy4 = tf.reduce_mean(tf.cast(top4_prediction, tf.float32))

print("Built model", flush=True)

# Load data ------------------------------------------------------------

print("Loading data", flush=True)

num_train_rows = 12000
num_test_rows = 1000
num_all_rows = num_train_rows + num_test_rows
prob_to_include_row = 0.05

all_input_data = np.zeros(shape=[num_all_rows]+input_shape)
all_target_data = np.zeros(shape=[num_all_rows]+target_shape)
all_target_data_weights = np.zeros(shape=[num_all_rows])

fill_features(num_all_rows, prob_to_include_row, all_input_data, all_target_data, all_target_data_weights)

print("Splitting into training and validation", flush=True)

indices = np.random.permutation(num_all_rows)
tinput_data = all_input_data[indices[:num_train_rows]]
vinput_data = all_input_data[indices[num_train_rows:]]
ttarget_data = all_target_data[indices[:num_train_rows]]
vtarget_data = all_target_data[indices[num_train_rows:]]
ttarget_data_weights = all_target_data_weights[indices[:num_train_rows]]
vtarget_data_weights = all_target_data_weights[indices[num_train_rows:]]

print("Data loading done", flush=True)

# Training ------------------------------------------------------------

num_epochs = 500

initial_sample_learning_rate = 0.0005
lr_plateau_decay_exponent = 3  #Exponent of the polynomial decay in learning rate based on number of plateaus
lr_plateau_decay_offset = 15   #Offset of the exponent
lr_plateau_no_better_epochs = 6 #Plateau if this many epochs with no training loss improvement
lr_plateau_min_epochs = 6       #And if at least this many epochs happened since the last plateau

batch_size = 10
def get_batches():
  idx = np.random.permutation(num_train_rows)
  batches = []
  num_batches = num_train_rows//batch_size
  for batchnum in range(num_batches):
    idx0 = idx[batchnum*batch_size : (batchnum+1)*batch_size]
    batches.append((tinput_data[idx0], ttarget_data[idx0], ttarget_data_weights[idx0]))
  return batches

print("Training", flush=True)


with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  def val_accuracy_and_loss():
    return session.run([accuracy1,accuracy4,loss], feed_dict={
      inputs: vinput_data,
      targets: vtarget_data,
      target_weights: vtarget_data_weights,
      batch_learning_rate: 0.0
    })

  (vacc1,vacc4,vloss) = val_accuracy_and_loss()
  print("Initial: vacc1 %f vacc4 %f vloss %f" % (vacc1, vacc4, vloss), flush=True)

  lr_best_epoch = 0
  lr_best_epoch_loss = None
  lr_reduction_count = 0
  lr_last_reduction_epoch = 0
  def sample_lr():
    return initial_sample_learning_rate / (((lr_reduction_count + lr_plateau_decay_offset) / lr_plateau_decay_offset) ** lr_plateau_decay_exponent)

  for epoch in range(num_epochs):
    batches = get_batches()
    num_batches = len(batches)

    tacc1_sum = 0
    tacc4_sum = 0
    tloss_sum = 0

    print("Epoch %d" % (epoch), end='', flush=True)

    for i in range(num_batches):
      (idata,tdata,tdataw) = batches[i]

      (bacc1, bacc4, bloss, _) = session.run(
        [accuracy1, accuracy4, loss, train_step],
        feed_dict={
          inputs: idata,
          targets: tdata,
          target_weights: tdataw,
          batch_learning_rate: sample_lr() * batch_size
        })

      tacc1_sum += bacc1
      tacc4_sum += bacc4
      tloss_sum += bloss

      if i % (num_batches // 30) == 0:
        print(".", end='', flush=True)

    print("")

    tacc1 = tacc1_sum / num_batches
    tacc4 = tacc4_sum / num_batches
    tloss = tloss_sum / num_batches
    (vacc1, vacc4,vloss) = val_accuracy_and_loss()

    if lr_best_epoch_loss is None or tloss < lr_best_epoch_loss:
      lr_best_epoch_loss = tloss
      lr_best_epoch = epoch

    if epoch >= lr_best_epoch + lr_plateau_no_better_epochs and epoch >= lr_last_reduction_epoch + lr_plateau_min_epochs:
      lr_last_reduction_epoch = epoch
      lr_reduction_count += 1

    print("tacc1 %f tacc4 %f tloss %f vacc1 %f vacc4 %f vloss %f lr %f" % (tacc1,tacc4,tloss,vacc1,vacc4,vloss,sample_lr()), flush=True)

  (vacc1,vacc4,vloss) = val_accuracy_and_loss()
  print("Final: vacc1 %f vacc4 %f vloss %f" % (vacc1, vacc4, vloss), flush=True)

  variables_names =[v.name for v in tf.trainable_variables()]
  values = session.run(variables_names)
  for k,v in zip(variables_names, values):
    print(k, v)
