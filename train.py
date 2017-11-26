#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import logging
import tensorflow as tf
import numpy as np

from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves

from board import Board

#Command and args-------------------------------------------------------------------

description = """
Train neural net on Go games!
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
parser.add_argument('-gamesdir', help='Dir of games to read', required=True, action='append')
parser.add_argument('-verbose', help='verbose', required=False, action='store_true')
args = vars(parser.parse_args())

traindir = args["traindir"]
gamesdirs = args["gamesdir"]
verbose = args["verbose"]

if not os.path.exists(traindir):
  os.makedirs(traindir)

bareformatter = logging.Formatter("%(message)s")
trainlogger = logging.getLogger("trainlogger")
trainlogger.setLevel(logging.INFO)
fh = logging.FileHandler(traindir+"/train.log", mode='w')
fh.setFormatter(bareformatter)
trainlogger.addHandler(fh)

detaillogger = logging.getLogger("detaillogger")
detaillogger.setLevel(logging.INFO)
fh = logging.FileHandler(traindir+"/detail.log", mode='w')
fh.setFormatter(bareformatter)
detaillogger.addHandler(fh)


#Test board --------------------------------------------------------------------
# board = Board(size=19)
# s = [123456789,787890901111]
# def rotl(x,k):
#   return ((x << k) | (x >> (64-k))) & 0xFFFFffffFFFFffff
# def rnext():
#   s0 = s[0]
#   s1 = s[1]
#   result = (s0+s1) & 0xFFFFffffFFFFffff
#   s1 ^= s0
#   s[0] = rotl(s0,55) ^ s1 ^ ((s1 << 14) & 0xFFFFffffFFFFffff)
#   s[1] = rotl(s1,36)
#   return result

# for i in range(1003500):
#   x = rnext() % 19
#   y = rnext() % 19
#   p = rnext() % 2 + 1
#   loc = board.loc(x,y)
#   if board.would_be_legal(p,loc):
#     board.play(p,loc)

# print(board.to_string())
# print(board.to_liberty_string(), flush=True)
# assert(False)

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

#TODO check if some KGS games leftmost variation actually isn't the game because of an undo
#TODO data symmetrizing
#TODO data deduplication
#TODO more data features?? definitely at least history
#TODO test different neural net structures, particularly the final combining layer
#TODO weight and neuron activation visualization
#TODO save weights and such, keep a notes file of configurations and results
#TODO run same NN several times to get an idea of consistency
#TODO does it help if we just enforce legality and don't need the NN to do so?
#TODO batch normalization
#TODO try residual structure?
#TODO gpu-acceleration!

max_board_size = 19
input_shape = [19,19,3]
target_shape = [19*19]

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
    # target_data[idx,max_board_size*max_board_size] = 1.0
  else:
    (y,x) = next_loc
    target_data[idx,y*max_board_size+x] = 1.0
    target_data_weights[idx] = 1.0

def fill_features(prob_to_include_row, input_data, target_data, target_data_weights, max_num_rows=None):
  idx = 0
  ngames = 0
  for filename in game_files:
    ngames += 1
    (slowboard,moves) = load_sgf_moves(filename)
    #TODO
    if slowboard.side != max_board_size:
      continue

    fastboard = Board(size=slowboard.side)
    for y in range(19):
      for x in range(19):
        loc = fastboard.loc(x,y)
        if slowboard.get(y,x) == 'b':
          fastboard.set_stone(Board.BLACK,loc)
        elif slowboard.get(y,x) == 'w':
          fastboard.set_stone(Board.WHITE,loc)

    for (color,next_loc) in moves:

      if random.random() < prob_to_include_row and \
         ((color == 'b') == (fastboard.pla == Board.BLACK)):
        if color == 'b':
          pla = 'b'
          opp = 'w'
        elif color == 'w':
          opp = 'b'
          pla = 'w'
        else:
          assert False

        if idx >= len(input_data):
          input_data.resize((idx * 3//2 + 100,) + input_data.shape[1:], refcheck=False)
          target_data.resize((idx * 3//2 + 100,) + target_data.shape[1:], refcheck=False)
          target_data_weights.resize((idx * 3//2 + 100,) + target_data_weights.shape[1:], refcheck=False)

        fill_row_features(slowboard,pla,opp,next_loc,input_data,target_data,target_data_weights,idx)
        idx += 1
        if max_num_rows is not None and idx >= max_num_rows:
          print("Loaded %d games and %d rows" % (ngames,idx), flush=True)
          input_data.resize((idx,) + input_data.shape[1:], refcheck=False)
          target_data.resize((idx,) + target_data.shape[1:], refcheck=False)
          target_data_weights.resize((idx,) + target_data_weights.shape[1:], refcheck=False)

          return
        if idx % 2500 == 0:
          print("Loaded %d games and %d rows" % (ngames,idx), flush=True)

      if next_loc is not None: # pass
        (row,col) = next_loc
        try:
          slowboard.play(row,col,color)
          loc = fastboard.loc(col,row)
          if color == 'b':
            fastboard.play(Board.BLACK,loc)
          else:
            fastboard.play(Board.WHITE,loc)

        except ValueError:
          print("Illegal move in: " + filename)
          print("Move " + str((row,col)))
          print(ascii_boards.render_board(slowboard))
          break

      else:
        fastboard.do_pass()

    for y in range(19):
      for x in range(19):
        loc = fastboard.loc(x,y)
        if slowboard.get(y,x) == 'b':
          assert(fastboard.board[loc] == Board.BLACK)
        elif slowboard.get(y,x) == 'w':
          assert(fastboard.board[loc] == Board.WHITE)
        elif slowboard.get(y,x) is None:
          assert(fastboard.board[loc] == Board.EMPTY)
        else:
          assert(False)


  print("Loaded %d games and %d rows" % (ngames,idx), flush=True)
  input_data.resize((idx,) + input_data.shape[1:], refcheck=False)
  target_data.resize((idx,) + target_data.shape[1:], refcheck=False)
  target_data_weights.resize((idx,) + target_data_weights.shape[1:], refcheck=False)

# Build model -------------------------------------------------------------

print("Building model", flush=True)

def init_stdev(num_inputs,num_outputs):
  #xavier
  #return math.sqrt(2.0 / (num_inputs + num_outputs))
  #herangzhen
  return math.sqrt(2.0 / (num_inputs))

def weight_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 1.0
  initial = tf.truncated_normal(shape=shape, stddev=stdev)
  return tf.Variable(initial,name=name)

def bias_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 2.0
  initial = tf.truncated_normal(shape=shape, mean=stdev, stddev=stdev)
  return tf.Variable(initial,name=name)

def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def reduce_stdev(x, axis=None, keepdims=False):
  m = tf.reduce_mean(x, axis=axis, keep_dims=True)
  devs_squared = tf.square(x - m)
  return tf.sqrt(tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims))

#Indexing:
#batch, bsize, bsize, channel

#Input layer
inputs = tf.placeholder(tf.float32, [None] + input_shape)

outputs_by_layer = []
cur_layer = inputs
cur_num_channels = input_shape[2]

#Convolutional RELU layer 1
conv1diam = 3
conv1num_channels = 32
conv1w = weight_variable("conv1w",[conv1diam,conv1diam,cur_num_channels,conv1num_channels],cur_num_channels*conv1diam**2,conv1num_channels)
conv1b = bias_variable("conv1b",[conv1num_channels],cur_num_channels,conv1num_channels)

cur_layer = tf.nn.relu(conv2d(cur_layer, conv1w) + conv1b)
cur_num_channels = conv1num_channels
outputs_by_layer.append(("conv1",cur_layer))

#Convolutional RELU layer 2
conv2diam = 3
conv2num_channels = 16
conv2w = weight_variable("conv2w",[conv2diam,conv2diam,cur_num_channels,conv2num_channels],cur_num_channels*conv2diam**2,conv2num_channels)
conv2b = bias_variable("conv2b",[conv2num_channels],cur_num_channels,conv2num_channels)

cur_layer = tf.nn.relu(conv2d(cur_layer, conv2w) + conv2b)
cur_num_channels = conv2num_channels
outputs_by_layer.append(("conv2",cur_layer))

#Convolutional RELU layer 3
conv3diam = 3
conv3num_channels = 8
conv3w = weight_variable("conv3w",[conv3diam,conv3diam,cur_num_channels,conv3num_channels],cur_num_channels*conv3diam**2,conv3num_channels)
conv3b = bias_variable("conv3b",[conv3num_channels],cur_num_channels,conv3num_channels)

cur_layer = tf.nn.relu(conv2d(cur_layer, conv3w) + conv3b)
cur_num_channels = conv3num_channels
outputs_by_layer.append(("conv3",cur_layer))

#Convolutional linear output layer 4
conv4diam = 5
conv4num_channels = 1
conv4w = weight_variable("conv4w",[conv4diam,conv4diam,cur_num_channels,conv4num_channels],cur_num_channels*conv4diam**2,conv4num_channels)

cur_layer = conv2d(cur_layer, conv4w)
cur_num_channels = conv4num_channels
outputs_by_layer.append(("conv4",cur_layer))

#Output
assert(cur_num_channels == 1)
output_layer = tf.reshape(cur_layer, [-1] + target_shape)

#Loss function
targets = tf.placeholder(tf.float32, [None] + target_shape)
target_weights = tf.placeholder(tf.float32, [None])
loss = tf.reduce_mean(target_weights * tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output_layer))

#Training results
batch_learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(batch_learning_rate).minimize(loss)
target_idxs = tf.argmax(targets, 1)
top1_prediction = tf.equal(tf.argmax(output_layer, 1), target_idxs)
top4_prediction = tf.nn.in_top_k(output_layer,target_idxs,4)
accuracy1 = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
accuracy4 = tf.reduce_mean(tf.cast(top4_prediction, tf.float32))

#Debugging stats
activated_prop_by_layer = [
  (name,tf.reduce_mean(tf.count_nonzero(layer,axis=[1,2])/max_board_size**2, axis=0)) for (name,layer) in outputs_by_layer
]
mean_output_by_layer = [
  (name,tf.reduce_mean(layer,axis=[0,1,2])) for (name,layer) in outputs_by_layer
]

stdev_output_by_layer = [
  (name,reduce_stdev(layer,axis=[0,1,2])**2) for (name,layer) in outputs_by_layer
]
mean_weights_by_var = [
  (v.name,tf.reduce_mean(v)) for v in tf.trainable_variables()
]
stdev_weights_by_var = [
  (v.name,reduce_stdev(v)) for v in tf.trainable_variables()
]

print("Built model", flush=True)

# Load data ------------------------------------------------------------

print("Loading data", flush=True)

prob_to_include_row = 0.05
all_input_data = np.zeros(shape=[1]+input_shape)
all_target_data = np.zeros(shape=[1]+target_shape)
all_target_data_weights = np.zeros(shape=[1])

max_num_rows = None
fill_features(prob_to_include_row, all_input_data, all_target_data, all_target_data_weights, max_num_rows = max_num_rows)

print("Splitting into training and validation", flush=True)
num_all_rows = len(all_input_data)
num_test_rows = min(5000,num_all_rows//10)
num_train_rows = num_all_rows - num_test_rows

indices = np.random.permutation(num_all_rows)
tinput_data = all_input_data[indices[:num_train_rows]]
vinput_data = all_input_data[indices[num_train_rows:]]
ttarget_data = all_target_data[indices[:num_train_rows]]
vtarget_data = all_target_data[indices[num_train_rows:]]
ttarget_data_weights = all_target_data_weights[indices[:num_train_rows]]
vtarget_data_weights = all_target_data_weights[indices[num_train_rows:]]
tdata = (tinput_data,ttarget_data,ttarget_data_weights)
vdata = (vinput_data,vtarget_data,vtarget_data_weights)

print("Data loading done", flush=True)

# Batching ------------------------------------------------------------

batch_size = 10

def get_batches():
  idx = np.random.permutation(num_train_rows)
  batches = []
  num_batches = num_train_rows//batch_size
  for batchnum in range(num_batches):
    idx0 = idx[batchnum*batch_size : (batchnum+1)*batch_size]
    batches.append((tinput_data[idx0], ttarget_data[idx0], ttarget_data_weights[idx0]))
  return batches

# Learning rate -------------------------------------------------------

class LR:
  def __init__(
    self,
    initial_lr,          #Initial learning rate by sample
    decay_exponent,      #Exponent of the polynomial decay in learning rate based on number of plateaus
    decay_offset,        #Offset of the exponent
    plateau_wait_epochs, #Plateau if this many epochs with no training loss improvement
    plateau_min_epochs   #And if at least this many epochs happened since the last plateau
  ):
    self.initial_lr = initial_lr
    self.decay_exponent = decay_exponent
    self.decay_offset = decay_offset
    self.plateau_wait_epochs = plateau_wait_epochs
    self.plateau_min_epochs = plateau_min_epochs

    self.best_epoch = 0
    self.best_epoch_loss = None
    self.reduction_count = 0
    self.last_reduction_epoch = 0

  def lr(self):
    factor = (self.reduction_count + self.decay_offset) / self.decay_offset
    return self.initial_lr / (factor ** self.decay_exponent)

  def report_loss(self,epoch,loss):
    if self.best_epoch_loss is None or loss < self.best_epoch_loss:
      self.best_epoch_loss = loss
      self.best_epoch = epoch

    if epoch >= self.best_epoch + self.plateau_wait_epochs and epoch >= self.last_reduction_epoch + self.plateau_min_epochs:
      self.last_reduction_epoch = epoch
      self.reduction_count += 1


# Training ------------------------------------------------------------

print("Training", flush=True)

num_epochs = 150

lr = LR(
  initial_lr = 0.0003,
  decay_exponent = 3,
  decay_offset = 15,
  plateau_wait_epochs = 4,
  plateau_min_epochs = 4,
)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  def run(fetches, data, blr=0.0):
    return session.run(fetches, feed_dict={
      inputs: data[0],
      targets: data[1],
      target_weights: data[2],
      batch_learning_rate: blr
    })

  def np_array_str(arr,precision):
    return np.array_str(arr, precision=precision, suppress_small = True, max_line_width = 200)

  def val_accuracy_and_loss():
    return run([accuracy1,accuracy4,loss], vdata)

  def train_stats_str(tacc1,tacc4,tloss):
    return "tacc1 %5.2f%% tacc4 %5.2f%% tloss %f" % (tacc1*100, tacc4*100, tloss)

  def validation_stats_str(vacc1,vacc4,vloss):
    return "vacc1 %5.2f%% vacc4 %5.2f%% vloss %f" % (vacc1*100, vacc4*100, vloss)

  def time_str(elapsed):
    return "time %.3f" % elapsed

  def log_detail_stats():
    apbl,mobl,sobl = run([dict(activated_prop_by_layer), dict(mean_output_by_layer), dict(stdev_output_by_layer)], vdata)
    for key in apbl:
      detaillogger.info("%s: activated_prop %s" % (key, np_array_str(apbl[key], precision=3)))
      detaillogger.info("%s: mean_output %s" % (key, np_array_str(mobl[key], precision=4)))
      detaillogger.info("%s: stdev_output %s" % (key, np_array_str(sobl[key], precision=4)))
      mw,sw = session.run([dict(mean_weights_by_var),dict(stdev_weights_by_var)])
    for key in mw:
      detaillogger.info("%s: mean weight %f stdev weight %f" % (key, mw[key], sw[key]))

  def run_batches(batches):
    num_batches = len(batches)

    tacc1_sum = 0
    tacc4_sum = 0
    tloss_sum = 0

    for i in range(num_batches):
      (bacc1, bacc4, bloss, _) = run(
        fetches=[accuracy1, accuracy4, loss, train_step],
        data=batches[i],
        blr=lr.lr() * batch_size
      )

      tacc1_sum += bacc1
      tacc4_sum += bacc4
      tloss_sum += bloss

      if i % (num_batches // 30) == 0:
        print(".", end='', flush=True)

    tacc1 = tacc1_sum / num_batches
    tacc4 = tacc4_sum / num_batches
    tloss = tloss_sum / num_batches
    return (tacc1,tacc4,tloss)

  (vacc1,vacc4,vloss) = val_accuracy_and_loss()
  vstr = validation_stats_str(vacc1,vacc4,vloss)

  print("Initial: %s" % (vstr), flush=True)
  trainlogger.info("Initial: %s" % (vstr))
  detaillogger.info("Initial: %s" % (vstr))
  log_detail_stats()

  start_time = time.perf_counter()
  for epoch in range(num_epochs):
    print("Epoch %d" % (epoch), end='', flush=True)
    batches = get_batches()
    (tacc1,tacc4,tloss) = run_batches(batches)
    (vacc1,vacc4,vloss) = val_accuracy_and_loss()
    lr.report_loss(epoch=epoch,loss=(tloss+vloss))
    print("")

    elapsed = time.perf_counter() - start_time

    tstr = train_stats_str(tacc1,tacc4,tloss)
    vstr = validation_stats_str(vacc1,vacc4,vloss)
    timestr = time_str(elapsed)
    print("%s %s lr %f %s" % (tstr,vstr,lr.lr(),timestr), flush=True)

    trainlogger.info("Epoch %d--------------------------------------------------" % (epoch))
    trainlogger.info("%s %s lr %f %s" % (tstr,vstr,lr.lr(),timestr))

    detaillogger.info("Epoch %d--------------------------------------------------" % (epoch))
    detaillogger.info("%s %s lr %f %s" % (tstr,vstr,lr.lr(),timestr))
    log_detail_stats()

  (vacc1,vacc4,vloss) = val_accuracy_and_loss()
  vstr = validation_stats_str(vacc1,vacc4,vloss)
  print("Final: %s" % (vstr), flush=True)
  trainlogger.info("Final: %s" % (vstr))
  detaillogger.info("Final: %s" % (vstr))

  variables_names =[v.name for v in tf.trainable_variables()]
  values = session.run(variables_names)
  for k,v in zip(variables_names, values):
    print(k, v)
