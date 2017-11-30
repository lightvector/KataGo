import logging
import math
import traceback
import tensorflow as tf
import numpy as np

from board import Board

#Feature extraction functions-------------------------------------------------------------------

#TODO data symmetrizing
#TODO data deduplication
#TODO test different neural net structures, particularly the final combining layer
#TODO weight and neuron activation visualization
#TODO run same NN several times to get an idea of consistency
#TODO does it help if we just enforce legality and don't need the NN to do so?
#TODO batch normalization
#TODO try residual structure?
#TODO gpu-acceleration!

#Neural net inputs
#19x19 is on board
#19x19 own stone present
#19x19 opp stone present
#19x19x2 own liberties 1,2
#19x19x2 opp liberties 1,2
#19x19x3 prev moves
#19x19x1 simple ko point

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

max_board_size = 19
input_shape = [19*19,13]
post_input_shape = [19,19,13]
target_shape = [19*19]
target_weights_shape = []
pass_pos = max_board_size * max_board_size

def xy_to_tensor_pos(x,y,offset):
  return (y+offset) * max_board_size + (x+offset)
def loc_to_tensor_pos(loc,board,offset):
  return (board.loc_y(loc) + offset) * max_board_size + (board.loc_x(loc) + offset)

def tensor_pos_to_loc(pos,board):
  if pos == pass_pos:
    return None
  bsize = board.size
  offset = (max_board_size - bsize) // 2
  x = pos%max_board_size - offset
  y = pos//max_board_size - offset
  if x < 0 or x >= bsize or y < 0 or y >= bsize:
    return board.loc(-1,-1) #Return an illegal move since this is offboard
  return board.loc(x,y)

def sym_tensor_pos(pos,symmetry):
  if pos == pass_pos:
    return pos
  x = pos%max_board_size
  y = pos//max_board_size
  if symmetry >= 4:
    symmetry -= 4
    tmp = x
    x = y
    y = tmp
  if symmetry >= 2:
    symmetry -= 2
    x = max_board_size-x-1
  if symmetry >= 1:
    symmetry -= 1
    y = max_board_size-y-1
  return y*max_board_size+x


def fill_row_features(board, pla, opp, moves, move_idx, input_data, target_data, target_data_weights, for_training, idx):
  bsize = board.size
  offset = (max_board_size - bsize) // 2
  for y in range(bsize):
    for x in range(bsize):
      pos = xy_to_tensor_pos(x,y,offset)
      input_data[idx,pos,0] = 1.0
      loc = board.loc(x,y)
      stone = board.board[loc]
      if stone == pla:
        input_data[idx,pos,1] = 1.0
        libs = board.num_liberties(loc)
        if libs == 1:
          input_data[idx,pos,3] = 1.0
        elif libs == 2:
          input_data[idx,pos,4] = 1.0
        elif libs == 3:
          input_data[idx,pos,5] = 1.0

      elif stone == opp:
        input_data[idx,pos,2] = 1.0
        libs = board.num_liberties(loc)
        if libs == 1:
          input_data[idx,pos,6] = 1.0
        elif libs == 2:
          input_data[idx,pos,7] = 1.0
        elif libs == 3:
          input_data[idx,pos,8] = 1.0

  if for_training:
    prob_to_include_prev1 = 0.90
    prob_to_include_prev2 = 0.95
    prob_to_include_prev3 = 0.95
  else:
    prob_to_include_prev1 = 1.00
    prob_to_include_prev2 = 1.00
    prob_to_include_prev3 = 1.00

  if move_idx >= 1 and np.random.random() < prob_to_include_prev1:
    prev1_loc = moves[move_idx-1][1]
    if prev1_loc is not None:
      pos = loc_to_tensor_pos(prev1_loc,board,offset)
      input_data[idx,pos,9] = 1.0

    if move_idx >= 2 and np.random.random() < prob_to_include_prev2:
      prev2_loc = moves[move_idx-2][1]
      if prev2_loc is not None:
        pos = loc_to_tensor_pos(prev2_loc,board,offset)
        input_data[idx,pos,10] = 1.0

      if move_idx >= 3 and np.random.random() < prob_to_include_prev3:
        prev3_loc = moves[move_idx-3][1]
        if prev3_loc is not None:
          pos = loc_to_tensor_pos(prev3_loc,board,offset)
          input_data[idx,pos,11] = 1.0

  if board.simple_ko_point is not None:
    pos = loc_to_tensor_pos(board.simple_ko_point,board,offset)
    input_data[idx,pos,12] = 1.0

  if target_data is not None:
    next_loc = moves[move_idx][1]
    if next_loc is None:
      # TODO for now we weight these rows to 0
      target_data[idx,0] = 1.0
      target_data_weights[idx] = 0.0
      pass
      # target_data[idx,max_board_size*max_board_size] = 1.0
    else:
      pos = loc_to_tensor_pos(next_loc,board,offset)
      target_data[idx,pos] = 1.0
      target_data_weights[idx] = 1.0


# Build model -------------------------------------------------------------

print("Building model", flush=True)

reg_variables = []
is_training = tf.placeholder(tf.bool)

def batchnorm(name,tensor):
  return tf.layers.batch_normalization(
    tensor,
    axis=-1, #Because channels are our last axis, -1 refers to that via wacky python indexing
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    training=is_training,
    name=name,
  )

def init_stdev(num_inputs,num_outputs):
  #xavier
  #return math.sqrt(2.0 / (num_inputs + num_outputs))
  #herangzhen
  return math.sqrt(2.0 / (num_inputs))

def weight_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 1.0
  initial = tf.truncated_normal(shape=shape, stddev=stdev)
  variable = tf.Variable(initial,name=name)
  reg_variables.append(variable)
  return variable

def bias_variable(name, shape, num_inputs, num_outputs):
  stdev = init_stdev(num_inputs,num_outputs) / 2.0
  initial = tf.truncated_normal(shape=shape, mean=stdev, stddev=stdev)
  variable = tf.Variable(initial,name=name)
  reg_variables.append(variable)
  return variable

def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

#Indexing:
#batch, bsize, bsize, channel

#Input layer
inputs = tf.placeholder(tf.float32, [None] + input_shape)

outputs_by_layer = []
cur_layer = tf.reshape(inputs, [-1] + post_input_shape)
cur_num_channels = post_input_shape[2]

#Convolutional RELU layer 1
conv1diam = 3
conv1num_channels = 64
conv1w = weight_variable("conv1w",[conv1diam,conv1diam,cur_num_channels,conv1num_channels],cur_num_channels*conv1diam**2,conv1num_channels)

cur_layer = tf.nn.relu(batchnorm("conv1norm",conv2d(cur_layer, conv1w)))
cur_num_channels = conv1num_channels
outputs_by_layer.append(("conv1",cur_layer))

#Convolutional RELU layer 2
conv2diam = 3
conv2num_channels = 32
conv2w = weight_variable("conv2w",[conv2diam,conv2diam,cur_num_channels,conv2num_channels],cur_num_channels*conv2diam**2,conv2num_channels)

cur_layer = tf.nn.relu(batchnorm("conv2norm",conv2d(cur_layer, conv2w)))
cur_num_channels = conv2num_channels
outputs_by_layer.append(("conv2",cur_layer))

#Convolutional RELU layer 3
conv3diam = 3
conv3num_channels = 32
conv3w = weight_variable("conv3w",[conv3diam,conv3diam,cur_num_channels,conv3num_channels],cur_num_channels*conv3diam**2,conv3num_channels)

cur_layer = tf.nn.relu(batchnorm("conv3norm",conv2d(cur_layer, conv3w)))
cur_num_channels = conv3num_channels
outputs_by_layer.append(("conv3",cur_layer))

#Convolutional RELU layer 4
conv4diam = 3
conv4num_channels = 32
conv4w = weight_variable("conv4w",[conv4diam,conv4diam,cur_num_channels,conv4num_channels],cur_num_channels*conv4diam**2,conv4num_channels)

cur_layer = tf.nn.relu(batchnorm("conv4norm",conv2d(cur_layer, conv4w)))
cur_num_channels = conv4num_channels
outputs_by_layer.append(("conv4",cur_layer))

#Convolutional linear output layer
convodiam = 5
convonum_channels = 1
convow = weight_variable("convow",[convodiam,convodiam,cur_num_channels,convonum_channels],cur_num_channels*convodiam**2,convonum_channels)

cur_layer = conv2d(cur_layer, convow)
cur_num_channels = convonum_channels
outputs_by_layer.append(("convo",cur_layer))

#Output
assert(cur_num_channels == 1)
output_layer = tf.reshape(cur_layer, [-1] + target_shape)

