import logging
import math
import traceback
import tensorflow as tf
import numpy as np

from board import Board

#Feature extraction functions-------------------------------------------------------------------

#TODO data deduplication
#TODO test different neural net structures, particularly the final combining layer
#TODO weight and neuron activation visualization
#TODO does it help if we just enforce legality and don't need the NN to do so?
#TODO try residual structure?
#TODO gpu-acceleration!

#Neural net inputs
#19x19 is on board
#19x19 own stone present
#19x19 opp stone present
#19x19x3 own liberties 1,2,3
#19x19x3 opp liberties 1,2,3
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

#Returns the new idx, which could be the same as idx if this isn't a good training row
def fill_row_features(board, pla, opp, moves, move_idx, input_data, target_data, target_data_weights, for_training, idx):
  if target_data is not None and moves[move_idx][1] is None:
    # TODO for now we skip passes
    return idx

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
      # TODO for now we skip passes
      return idx
      # target_data[idx,max_board_size*max_board_size] = 1.0
    else:
      pos = loc_to_tensor_pos(next_loc,board,offset)
      target_data[idx,pos] = 1.0
      target_data_weights[idx] = 1.0

  return idx+1


# Build model -------------------------------------------------------------

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

def weight_variable_init_zero(name, shape):
  initial = tf.constant_initializer(0.0)
  variable = tf.Variable(initial,name=name)
  reg_variables.append(variable)
  return variable

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

def apply_symmetry(tensor,symmetries,inverse):
  ud = symmetries[0]
  lr = symmetries[1]
  transp = symmetries[2]

  rev_axes = tf.concat([
    tf.cond(ud, lambda: tf.constant([1]), lambda: tf.constant([],dtype='int32')),
    tf.cond(lr, lambda: tf.constant([2]), lambda: tf.constant([],dtype='int32')),
  ], axis=0)

  if not inverse:
    tensor = tf.reverse(tensor, rev_axes)

  assert(len(tensor.shape) == 4)
  tensor = tf.cond(
    transp,
    lambda: tf.transpose(tensor, [0,2,1,3]),
    lambda: tensor)

  if inverse:
    tensor = tf.reverse(tensor, rev_axes)

  return tensor

#Define useful components --------------------------------------------------------------------------

#Accumulates outputs for printing stats about their activations
outputs_by_layer = []

def parametric_relu(name, layer):
  assert(len(layer.shape) == 4)
  num_channels = layer.shape[3].value
  alphas = weight_variable_init_zero(name+"/prelu",[1,1,1,num_channels])
  return tf.maximum(0.0,layer) + alphas * tf.minimum(0.0,layer)

#Convolutional layer with batch norm and nonlinear activation
def conv_block(name, in_layer, diam, in_channels, out_channels):
  weights = weight_variable(name+"/w",[diam,diam,in_channels,out_channels],in_channels*diam*diam,out_channels)
  out_layer = parametric_relu("/prelu",batchnorm(name+"/norm",conv2d(in_layer, weights)))
  outputs_by_layer.append((name,out_layer))
  return out_layer

#Convoution only, no batch norm or nonlinearity
def conv_only_block(name, in_layer, diam, in_channels, out_channels):
  weights = weight_variable(name+"/w",[diam,diam,in_channels,out_channels],in_channels*diam*diam,out_channels)
  out_layer = conv2d(in_layer, weights)
  outputs_by_layer.append((name,out_layer))
  return out_layer

#Convolutional residual block with internal batch norm and nonlinear activation
def res_conv_block(name, in_layer, diam, main_channels, mid_channels):
  trans1_layer = parametric_relu("/prelu1",(batchnorm(name+"/norm1",in_layer))
  outputs_by_layer.append((name+"/trans1",trans1_layer))

  weights1 = weight_variable(name+"/w1",[diam,diam,main_channels,mid_channels],main_channels*diam*diam,mid_channels)
  conv1_layer = conv2d(trans1_layer, weights1)
  outputs_by_layer.append((name+"/conv1",conv1_layer))

  trans2_layer = parametric_relu("/prelu2",(batchnorm(name+"/norm2",conv1_layer))
  outputs_by_layer.append((name+"/trans2",trans2_layer))

  weights2 = weight_variable(name+"/w2",[diam,diam,mid_channels,main_channels],mid_channels*diam*diam,main_channels)
  conv2_layer = conv2d(trans2_layer, weights2)
  outputs_by_layer.append((name+"/conv2",conv2_layer))

  residual = conv2_layer
  out_layer = in_layer + residual
  outputs_by_layer.append((name,out_layer))
  return out_layer


#Special block for detecting ladders, with mid_channels channels per each of 4 diagonal scans.
def ladder_block(name, in_layer, main_channels, mid_channels):
  # Converts [[123][456][789]] to [[12300][04560][00789]]
  def skew_right(tensor):
    n = max_board_size
    assert(tensor.shape[1].value == n)
    assert(tensor.shape[2].value == n)
    c = tensor.shape[3].value
    tensor = tf.pad(tensor,[[0,0],[0,0],[0,n],[0,0]]) #Pad 19x19 -> 19x38
    tensor = tf.reshape(tensor,[-1,2*n*n,c]) #Linearize
    tensor = tensor[:,:((2*n-1)*n),:] #Chop off the 19 zeroes on the end
    tensor = tf.reshape(tensor,[-1,n,2*n-1,c]) #Now we are skewed 19x37 as desired
    return tensor
  # Converts [[12345][6789a][bcdef]] to [[123][789][def]]
  def unskew_right(tensor):
    n = max_board_size
    assert(tensor.shape[1].value == n)
    assert(tensor.shape[2].value == 2*n-1)
    c = tensor.shape[3].value
    tensor = tf.reshape(tensor,[-1,n*(2*n-1),c]) #Linearize
    tensor = tf.pad(tensor,[[0,0],[0,n],[0,0]]) #Pad 19*37 -> 19*38
    tensor = tf.reshape(tensor,[-1,n,2*n,c]) #Convert back to 19x38
    tensor = tensor[:,:,:n,:] #Chop off the extra, now we are 19x19
    return tensor

  # Converts [[123][456][789]] to [[00123][04560][78900]]
  def skew_left(tensor):
    n = max_board_size
    assert(tensor.shape[1].value == n)
    assert(tensor.shape[2].value == n)
    c = tensor.shape[3].value
    tensor = tf.pad(tensor,[[0,0],[1,1],[n-2,0],[0,0]]) #Pad 19x19 -> 21x36
    tensor = tf.reshape(tensor,[-1,(n+2)*(2*n-2),c]) #Linearize
    tensor = tensor[:,(2*n-3):(-n+1),:] #Chop off the 35 extra zeroes on the start and the 18 at the end.
    tensor = tf.reshape(tensor,[-1,n,2*n-1,c]) #Now we are skewed 19x37 as desired
    return tensor

  # Converts [[12345][6789a][bcdef]] to [[345][789][bcd]]
  def unskew_left(tensor):
    n = max_board_size
    assert(tensor.shape[1].value == n)
    assert(tensor.shape[2].value == 2*n-1)
    c = tensor.shape[3].value
    tensor = tf.reshape(tensor,[-1,n*(2*n-1),c]) #Linearize
    tensor = tf.pad(tensor,[[0,0],[2*n-3,n-1],[0,0]]) #Pad 19*37 -> 21*36
    tensor = tf.reshape(tensor,[-1,n+2,2*n-2,c]) #Convert back to 21x36
    tensor = tensor[:,1:(n+1),(n-2):,:] #Chop off the extra, now we are 19x19
    return tensor

  #First, as usual, batchnorm and relu the trunk to get the values to a reasonable scale
  trans1_layer = parametric_relu("/prelu1",(batchnorm(name+"/norm1",in_layer))
  outputs_by_layer.append((name+"/trans1",trans1_layer))

  c = mid_channels

  #The next part basically does a scan across the board each of the 4 diagonal ways, computing a moving average.
  #We use a convolution to let the neural net choose the values and weights:
  #a: value on this spot to be moving-averaged
  #b: if the weight on the moving average so far is 1, the value on this spot gets a factor of exp(b)-1 weight.
  diampre = 3
  weightsprea = weight_variable(name+"/wprea",[diampre,diampre,main_channels,mid_channels*4],main_channels*diampre*diampre,c*4)
  weightspreb = weight_variable(name+"/wpreb",[diampre,diampre,main_channels,mid_channels*4],main_channels*diampre*diampre,c*4)

  convprea_layer = conv2d(trans1_layer, weightsprea)
  convpreb_layer = conv2d(trans1_layer, weightspreb)
  outputs_by_layer.append((name+"/convprea",convprea_layer))
  outputs_by_layer.append((name+"/convpreb",convpreb_layer))

  transprea_layer = parametric_relu("/preluprea",(batchnorm(name+"/normprea",convprea_layer))
  transpreb_layer = tf.nn.sigmoid(batchnorm(name+"/normpreb",convpreb_layer)) * 1.5 + 0.0001
  outputs_by_layer.append((name+"/transprea",transprea_layer))
  outputs_by_layer.append((name+"/transpreb",transpreb_layer))

  #Now, skew each segment of the channels left and right, so that axis=1 now runs diagonally along the original board
  skewed_r_a = skew_right(transprea_layer[:,:,:,:(2*c)])
  skewed_r_b = skew_right(transpreb_layer[:,:,:,:(2*c)])
  skewed_l_a = skew_left(transprea_layer[:,:,:,(2*c):])
  skewed_l_b = skew_left(transpreb_layer[:,:,:,(2*c):])

  #And extract out all the necessary bits
  r_fwd_a = skewed_r_a[:,:,:,:c]
  r_rev_a = skewed_r_a[:,:,:,c:]
  r_fwd_b = skewed_r_b[:,:,:,:c]
  r_rev_b = skewed_r_b[:,:,:,c:]

  l_fwd_a = skewed_l_a[:,:,:,:c]
  l_rev_a = skewed_l_a[:,:,:,c:]
  l_fwd_b = skewed_l_b[:,:,:,:c]
  l_rev_b = skewed_l_b[:,:,:,c:]

  #Compute the proper weights based on b
  r_fwd_bsum = tf.cumsum(r_fwd_b, axis=1, exclusive=True)
  r_rev_bsum = tf.cumsum(r_rev_b, axis=1, exclusive=True, reverse=True)
  l_fwd_bsum = tf.cumsum(l_fwd_b, axis=1, exclusive=True)
  l_rev_bsum = tf.cumsum(l_rev_b, axis=1, exclusive=True, reverse=True)
  r_fwd_weight = tf.exp(r_fwd_b+r_fwd_bsum) - tf.exp(r_fwd_bsum)
  r_rev_weight = tf.exp(r_rev_b+r_rev_bsum) - tf.exp(r_rev_bsum)
  l_fwd_weight = tf.exp(l_fwd_b+l_fwd_bsum) - tf.exp(l_fwd_bsum)
  l_rev_weight = tf.exp(l_rev_b+l_rev_bsum) - tf.exp(l_rev_bsum)

  #Compute the moving averages
  result_r_fwd = tf.cumsum(r_fwd_a * r_fwd_weight, axis=1              ) / tf.cumsum(r_fwd_weight, axis=1)
  result_r_rev = tf.cumsum(r_rev_a * r_rev_weight, axis=1, reverse=True) / tf.cumsum(r_rev_weight, axis=1, reverse=True)
  result_l_fwd = tf.cumsum(l_fwd_a * l_fwd_weight, axis=1              ) / tf.cumsum(l_fwd_weight, axis=1)
  result_l_rev = tf.cumsum(l_rev_a * l_rev_weight, axis=1, reverse=True) / tf.cumsum(l_rev_weight, axis=1, reverse=True)

  #Unskew concatenate everything back together
  results = [unskew_right(result_r_fwd), unskew_right(result_r_rev), unskew_left(result_l_fwd), unskew_left(result_l_rev)]
  results = tf.concat(results,axis=3)

  #Apply a convolution to merge the result back into the trunk
  diampost = 1
  weightspost = weight_variable(name+"/wpost",[diampost,diampost,mid_channels*4,main_channels],mid_channels*4*diampost*diampost,main_channels)
  convpost_layer = conv2d(results, weightspost)
  outputs_by_layer.append((name+"/convpost",convpost_layer))

  residual = convpost_layer
  out_layer = in_layer + residual
  outputs_by_layer.append((name,out_layer))
  return out_layer


#Begin Neural net------------------------------------------------------------------------------------

#Indexing:
#batch, bsize, bsize, channel

#Input layer---------------------------------------------------------------------------------
inputs = tf.placeholder(tf.float32, [None] + input_shape)
symmetries = tf.placeholder(tf.bool, [3])

cur_layer = tf.reshape(inputs, [-1] + post_input_shape)
input_num_channels = post_input_shape[2]
#Input symmetries - we apply symmetries during training by transforming the input and reverse-transforming the output
cur_layer = apply_symmetry(cur_layer,symmetries,inverse=False)

#Convolutional RELU layer 1-------------------------------------------------------------------------------------
cur_layer = conv_only_block("conv1",cur_layer,diam=5,in_channels=input_num_channels,out_channels=192)

#Residual Convolutional Block 1---------------------------------------------------------------------------------
cur_layer = res_conv_block("rconv1",cur_layer,diam=3,main_channels=192,mid_channels=192)

#Residual Convolutional Block 2---------------------------------------------------------------------------------
cur_layer = res_conv_block("rconv2",cur_layer,diam=3,main_channels=192,mid_channels=192)

#Residual Convolutional Block 3---------------------------------------------------------------------------------
cur_layer = res_conv_block("rconv3",cur_layer,diam=3,main_channels=192,mid_channels=192)

#Residual Convolutional Block 4---------------------------------------------------------------------------------
cur_layer = res_conv_block("rconv4",cur_layer,diam=3,main_channels=192,mid_channels=192)

#Postprocessing residual trunk----------------------------------------------------------------------------------

#Normalize and relu just before the policy head
cur_layer = parametric_relu("/prelutrunk",(batchnorm("normtrunk",cur_layer))
outputs_by_layer.append(("trunk",cur_layer))

#Policy head---------------------------------------------------------------------------------
p0_layer = cur_layer

#This is the main path for policy information
p1_num_channels = 48
p1_intermediate_conv = conv_only_block("p1/intermediate_conv",p0_layer,diam=3,in_channels=192,out_channels=p1_num_channels)

#But in parallel convolve to compute some features about the global state of the board
#Hopefully the neural net uses this for stuff like ko situation, overall temperature/threatyness, who is leading, etc.
g1_num_channels = 16
g1_layer = conv_block("g1",p0_layer,diam=3,in_channels=192,out_channels=g1_num_channels)

#Fold g1 down to single values for the board.
#For stdev, add a tiny constant to ensure numeric stability
g1_mean = tf.reduce_mean(g1_layer,axis=[1,2],keep_dims=True)
g1_max = tf.reduce_max(g1_layer,axis=[1,2],keep_dims=True)
g1_stdev = tf.sqrt(tf.reduce_mean(tf.square(g1_layer - g1_mean), axis=[1,2], keep_dims=True) + (1e-4))
g2_layer = tf.concat([g1_mean,g1_max,g1_stdev],axis=3) #shape [b,1,1,3*convg1num_channels]
g2_num_channels = 3*g1_num_channels
outputs_by_layer.append(("g2",g2_layer))

#Transform them into the space of the policy features to act as biases for the policy
#Also divide the initial weights a bit more because we think these should matter a bit less than local shape stuff,
#by multiplying the number of inputs for purposes of weight initialization (currently mult by 4)
matmulg2w = weight_variable("matmulg2w",[g2_num_channels,p1_num_channels],g2_num_channels*4,p1_num_channels)
g3_layer = tf.tensordot(g2_layer,matmulg2w,axes=[[3],[0]])
outputs_by_layer.append(("g3",g3_layer))

#Add! This adds shapes [b,19,19,convp1_num_channels] + [b,1,1,convp1_num_channels]
#so the second one should get broadcast up to the size of the first one.
#We can think of p1 as being an ordinary convolution layer except that for every node of the convolution, the g2 values (g2_num_channels many of them)
#have been appended to the p0 incoming values (p0_num_channels * convp1diam * convp1diam many of them).
#The matrix matmulg2w is simply the set of weights for that additional part of the matrix. It's just that rather than appending beforehand,
#we multiply separately and add to the output afterward.
p1_intermediate_sum = p1_intermediate_conv + g3_layer

#And now apply batchnorm and crelu
p1_layer = tf.nn.crelu(batchnorm("p1/norm",p1_intermediate_sum))
outputs_by_layer.append(("p1",p1_layer))

#Finally, apply linear convolution to produce final output
#96 in_channels due to crelu (48 x 2)
p2_layer = conv_only_block("p2",p1_layer,diam=5,in_channels=96,out_channels=1)

#Output symmetries - we apply symmetries during training by transforming the input and reverse-transforming the output
policy_output = apply_symmetry(p2_layer,symmetries,inverse=True)
policy_output = tf.reshape(policy_output, [-1] + target_shape)

#Add pass move based on the global g values
#matmulpass = weight_variable("matmulpass",[pass_num_channels,1],g2_num_channels,1)
#pass_output = tf.matmul(g2_output,matmulpass)
#outputs_by_layer.append(("pass",pass_output))
#policy_output = tf.concat([policy_output,pass_output],axis=1)

