import logging
import math
import traceback
import tensorflow as tf
import numpy as np

from board import Board

#Feature extraction functions-------------------------------------------------------------------

class Model:

  def __init__(self,config):
    self.max_board_size = 19
    self.num_input_features = 19
    self.input_shape = [19*19,self.num_input_features]
    self.post_input_shape = [19,19,self.num_input_features]
    self.policy_target_shape_nopass = [19*19]
    self.policy_target_shape = [19*19+1] #+1 for pass move
    self.value_target_shape = []
    self.target_weights_shape = []
    self.rank_shape=[1+9+(17+9)+(19+9)]
    self.rank_embedding_dim = 8

    self.pass_pos = self.max_board_size * self.max_board_size

    self.reg_variables = []
    self.lr_adjusted_variables = {}
    self.is_training = tf.placeholder(tf.bool)

    #Accumulates outputs for printing stats about their activations
    self.outputs_by_layer = []

    use_ranks = config["use_ranks"]
    include_policy = config["include_policy"]
    include_value = config["include_value"]
    predict_pass = config["predict_pass"]
    self.build_model(use_ranks, include_policy, include_value, predict_pass)

  def xy_to_tensor_pos(self,x,y,offset):
    return (y+offset) * self.max_board_size + (x+offset)
  def loc_to_tensor_pos(self,loc,board,offset):
    return (board.loc_y(loc) + offset) * self.max_board_size + (board.loc_x(loc) + offset)

  def tensor_pos_to_loc(self,pos,board):
    if pos == self.pass_pos:
      return None
    max_board_size = self.max_board_size
    bsize = board.size
    offset = (max_board_size - bsize) // 2
    x = pos % max_board_size - offset
    y = pos // max_board_size - offset
    if x < 0 or x >= bsize or y < 0 or y >= bsize:
      return board.loc(-1,-1) #Return an illegal move since this is offboard
    return board.loc(x,y)

  def sym_tensor_pos(self,pos,symmetry):
    if pos == self.pass_pos:
      return pos
    max_board_size = self.max_board_size
    x = pos % max_board_size
    y = pos // max_board_size
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
    return y * max_board_size + x

  #Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
  def iterLadders(self, board, f):
    chainHeadsSolved = {}
    copy = board.copy()

    bsize = board.size
    offset = (self.max_board_size - bsize) // 2

    for y in range(bsize):
      for x in range(bsize):
        pos = self.xy_to_tensor_pos(x,y,offset)
        loc = board.loc(x,y)
        stone = board.board[loc]

        if stone == Board.BLACK or stone == Board.WHITE:
          libs = board.num_liberties(loc)
          if libs == 1 or libs == 2:
            head = board.group_head[loc]
            if head in chainHeadsSolved:
              laddered = chainHeadsSolved[head]
              if laddered:
                f(loc,pos,[])
            else:
              #Perform search on copy so as not to mess up tracking of solved heads
              if libs == 1:
                workingMoves = []
                laddered = copy.searchIsLadderCaptured(loc,True)
              else:
                workingMoves = copy.searchIsLadderCapturedAttackerFirst2Libs(loc)
                laddered = len(workingMoves) > 0

              chainHeadsSolved[head] = laddered
              if laddered:
                f(loc,pos,workingMoves)


  #Returns the new idx, which could be the same as idx if this isn't a good training row
  def fill_row_features(self, board, pla, opp, moves, move_idx, input_data, self_komi, use_history_prop, idx):
    bsize = board.size
    offset = (self.max_board_size - bsize) // 2

    for y in range(bsize):
      for x in range(bsize):
        pos = self.xy_to_tensor_pos(x,y,offset)
        input_data[idx,pos,0] = 1.0
        input_data[idx,pos,18] = self_komi / 15.0;
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

    if board.simple_ko_point is not None:
      pos = self.loc_to_tensor_pos(board.simple_ko_point,board,offset)
      input_data[idx,pos,9] = 1.0

    if use_history_prop > 0.0:
      if move_idx >= 1 and moves[move_idx-1][0] == opp:
        prev1_loc = moves[move_idx-1][1]
        if prev1_loc is not None:
          pos = self.loc_to_tensor_pos(prev1_loc,board,offset)
          input_data[idx,pos,10] = use_history_prop

        if move_idx >= 2 and moves[move_idx-2][0] == pla:
          prev2_loc = moves[move_idx-2][1]
          if prev2_loc is not None:
            pos = self.loc_to_tensor_pos(prev2_loc,board,offset)
            input_data[idx,pos,11] = use_history_prop

          if move_idx >= 3 and moves[move_idx-3][0] == opp:
            prev3_loc = moves[move_idx-3][1]
            if prev3_loc is not None:
              pos = self.loc_to_tensor_pos(prev3_loc,board,offset)
              input_data[idx,pos,12] = use_history_prop

            if move_idx >= 4 and moves[move_idx-4][0] == pla:
              prev4_loc = moves[move_idx-4][1]
              if prev4_loc is not None:
                pos = self.loc_to_tensor_pos(prev4_loc,board,offset)
                input_data[idx,pos,13] = use_history_prop

              if move_idx >= 5 and moves[move_idx-5][0] == opp:
                prev5_loc = moves[move_idx-5][1]
                if prev5_loc is not None:
                  pos = self.loc_to_tensor_pos(prev5_loc,board,offset)
                  input_data[idx,pos,14] = use_history_prop

    def addLadderFeature(loc,pos,workingMoves):
      assert(board.board[loc] == Board.BLACK or board.board[loc] == Board.WHITE);
      libs = board.num_liberties(loc)
      if libs == 1:
        input_data[idx,pos,15] = 1.0
      else:
        input_data[idx,pos,16] = 1.0
        for workingMove in workingMoves:
          workingPos = self.loc_to_tensor_pos(workingMove,board,offset)
          input_data[idx,workingPos,17] = 1.0

    self.iterLadders(board, addLadderFeature)

    return idx+1


  # Build model -------------------------------------------------------------

  def ensure_variable_exists(self,name):
    for v in tf.trainable_variables():
      if v.name == name:
        return name
    raise Exception("Could not find variable " + name)

  def add_lr_factor(self,name,factor):
    self.ensure_variable_exists(name)
    if name in self.lr_adjusted_variables:
      self.lr_adjusted_variables[name] = factor * self.lr_adjusted_variables[name]
    else:
      self.lr_adjusted_variables[name] = factor

  def batchnorm(self,name,tensor):
    return tf.layers.batch_normalization(
      tensor,
      axis=-1, #Because channels are our last axis, -1 refers to that via wacky python indexing
      momentum=0.99,
      epsilon=0.001,
      center=True,
      scale=False,
      training=self.is_training,
      name=name,
    )

  def init_stdev(self,num_inputs,num_outputs):
    #xavier
    #return math.sqrt(2.0 / (num_inputs + num_outputs))
    #herangzhen
    return math.sqrt(2.0 / (num_inputs))

  def init_weights(self, shape, num_inputs, num_outputs):
    stdev = self.init_stdev(num_inputs,num_outputs) / 1.0
    return tf.truncated_normal(shape=shape, stddev=stdev)

  def weight_variable_init_constant(self, name, shape, constant):
    init = tf.zeros(shape)
    if constant != 0.0:
      init = init + constant
    variable = tf.Variable(init,name=name)
    self.reg_variables.append(variable)
    return variable

  def weight_variable(self, name, shape, num_inputs, num_outputs, scale_initial_weights=1.0, extra_initial_weight=None, reg=True):
    initial = self.init_weights(shape, num_inputs, num_outputs)
    if extra_initial_weight is not None:
      initial = initial + extra_initial_weight
    initial = initial * scale_initial_weights

    variable = tf.Variable(initial,name=name)
    if reg:
      self.reg_variables.append(variable)
    return variable

  def conv2d(self, x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

  def dilated_conv2d(self, x, w, dilation):
    return tf.nn.atrous_conv2d(x, w, rate = dilation, padding='SAME')

  def apply_symmetry(self,tensor,symmetries,inverse):
    ud = symmetries[0]
    lr = symmetries[1]
    transp = symmetries[2]

    rev_axes = tf.concat([
      tf.cond(ud, lambda: tf.constant([1]), lambda: tf.constant([],dtype='int32')),
      tf.cond(lr, lambda: tf.constant([2]), lambda: tf.constant([],dtype='int32')),
    ], axis=0)

    if not inverse:
      tensor = tf.reverse(tensor, rev_axes)

    assert(len(tensor.shape) == 4 or len(tensor.shape) == 3)
    if len(tensor.shape) == 3:
      tensor = tf.cond(
        transp,
        lambda: tf.transpose(tensor, [0,2,1]),
        lambda: tensor)
    else:
      tensor = tf.cond(
        transp,
        lambda: tf.transpose(tensor, [0,2,1,3]),
        lambda: tensor)

    if inverse:
      tensor = tf.reverse(tensor, rev_axes)

    return tensor


  def chain_pool(self,tensor,chains,num_chain_segments,empty,nonempty,mode):
    bsize = self.max_board_size
    assert(len(tensor.shape) == 4)
    assert(len(chains.shape) == 3)
    assert(len(num_chain_segments.shape) == 1)
    assert(tensor.shape[1].value == bsize)
    assert(tensor.shape[2].value == bsize)
    assert(chains.shape[1].value == bsize)
    assert(chains.shape[2].value == bsize)
    assert(mode == "sum" or mode == "max")
    num_channels = tensor.shape[3].value

    #Since tf.unsorted_segment* doesn't operate by batches or channels, we need to manually construct
    #a different shift to add to each batch and each channel so that they pool into disjoint buckets.
    #Each one needs max_chain_idxs different buckets.
    num_segments_by_batch_and_channel = tf.fill([1,num_channels],1) * tf.expand_dims(num_chain_segments,axis=1)
    shift = tf.cumsum(tf.reshape(num_segments_by_batch_and_channel,[-1]),exclusive=True)
    num_segments = tf.reduce_sum(num_chain_segments) * num_channels
    shift = tf.reshape(shift,[-1,1,1,num_channels])

    segments = tf.expand_dims(chains,3) + shift
    if mode == "sum":
      pools = tf.unsorted_segment_sum(tensor,segments,num_segments=num_segments)
    elif mode == "max":
      pools = tf.unsorted_segment_max(tensor,segments,num_segments=num_segments)
    else:
      assert False

    gathered = tf.gather(pools,indices=segments)
    return gathered * tf.expand_dims(nonempty,axis=3) # + tensor * empty


  manhattan_radius_3_kernel = tf.reshape(tf.constant([
    [0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]
  ], dtype=tf.float32), [7,7,1,1])

  #Define useful components --------------------------------------------------------------------------

  def parametric_relu(self, name, layer):
    assert(len(layer.shape) == 4)
    #num_channels = layer.shape[3].value
    #alphas = self.weight_variable_init_constant(name+"/prelu",[1,1,1,num_channels],constant=0.0)
    return tf.nn.relu(layer)

  def parametric_relu_non_spatial(self, name, layer):
    assert(len(layer.shape) == 2)
    #num_channels = layer.shape[1].value
    #alphas = self.weight_variable_init_constant(name+"/prelu",[1,num_channels],constant=0.0)
    return tf.nn.relu(layer)

  def merge_residual(self,name,trunk,residual):
    trunk = trunk + residual
    self.outputs_by_layer.append((name,trunk))
    return trunk

  def conv_weight_variable(self, name, diam1, diam2, in_channels, out_channels, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None, reg=True):
    radius1 = diam1 // 2
    radius2 = diam2 // 2

    if emphasize_center_weight is None:
      weights = self.weight_variable(name,[diam1,diam2,in_channels,out_channels],in_channels*diam1*diam2,out_channels,scale_initial_weights,reg=reg)
    else:
      extra_initial_weight = self.init_weights([1,1,in_channels,out_channels], in_channels, out_channels) * emphasize_center_weight
      extra_initial_weight = tf.pad(extra_initial_weight, [(radius1,radius1),(radius2,radius2),(0,0),(0,0)])
      weights = self.weight_variable(name,[diam1,diam2,in_channels,out_channels],in_channels*diam1*diam2,out_channels,scale_initial_weights,extra_initial_weight,reg=reg)

    if emphasize_center_lr is not None:
      factor = tf.constant([emphasize_center_lr],dtype=tf.float32)
      factor = tf.reshape(factor,[1,1,1,1])
      factor = tf.pad(factor, [(radius1,radius1),(radius2,radius2),(0,0),(0,0)], constant_values=1.0)
      self.add_lr_factor(weights.name, factor)

    return weights

  #Convolutional layer with batch norm and nonlinear activation
  def conv_block(self, name, in_layer, diam, in_channels, out_channels, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None):
    weights = self.conv_weight_variable(name+"/w", diam, diam, in_channels, out_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    out_layer = self.parametric_relu(name+"/prelu",self.batchnorm(name+"/norm",self.conv2d(in_layer, weights)))
    self.outputs_by_layer.append((name,out_layer))
    return out_layer

  #Convolution only, no batch norm or nonlinearity
  def conv_only_block(self, name, in_layer, diam, in_channels, out_channels, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None, reg=True):
    weights = self.conv_weight_variable(name+"/w", diam, diam, in_channels, out_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr, reg=reg)
    out_layer = self.conv2d(in_layer, weights)
    self.outputs_by_layer.append((name,out_layer))
    return out_layer

  #Convolution emphasizing the center
  def conv_only_extra_center_block(self, name, in_layer, diam, in_channels, out_channels, scale_initial_weights=1.0):
    radius = diam // 2
    center_weights = self.weight_variable(name+"/wcenter",[1,1,in_channels,out_channels],in_channels,out_channels,scale_initial_weights=0.3*scale_initial_weights)
    weights = self.weight_variable(name+"/w",[diam,diam,in_channels,out_channels],in_channels*diam*diam,out_channels,scale_initial_weights)
    weights = weights + tf.pad(center_weights,[(radius,radius),(radius,radius),(0,0),(0,0)])
    out_layer = self.conv2d(in_layer, weights)
    self.outputs_by_layer.append((name,out_layer))
    return out_layer

  #Convolutional residual block with internal batch norm and nonlinear activation
  def res_conv_block(self, name, in_layer, diam, main_channels, mid_channels, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None):
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    weights1 = self.conv_weight_variable(name+"/w1", diam, diam, main_channels, mid_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    conv1_layer = self.conv2d(trans1_layer, weights1)
    self.outputs_by_layer.append((name+"/conv1",conv1_layer))

    trans2_layer = self.parametric_relu(name+"/prelu2",(self.batchnorm(name+"/norm2",conv1_layer)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels, main_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer

  #Convolutional residual block with internal batch norm and nonlinear activation
  def global_res_conv_block(self, name, in_layer, diam, main_channels, mid_channels, global_mid_channels, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None):
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    weights1a = self.conv_weight_variable(name+"/w1a", diam, diam, main_channels, mid_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    weights1b = self.conv_weight_variable(name+"/w1b", diam, diam, main_channels, global_mid_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    conv1a_layer = self.conv2d(trans1_layer, weights1a)
    conv1b_layer = self.conv2d(trans1_layer, weights1b)
    self.outputs_by_layer.append((name+"/conv1a",conv1a_layer))
    self.outputs_by_layer.append((name+"/conv1b",conv1b_layer))

    trans1b_layer = self.parametric_relu(name+"/trans1b",(self.batchnorm(name+"/norm1b",conv1b_layer)))
    trans1b_mean = tf.reduce_mean(trans1b_layer,axis=[1,2],keepdims=True)
    trans1b_max = tf.reduce_max(trans1b_layer,axis=[1,2],keepdims=True)
    trans1b_pooled = tf.concat([trans1b_mean,trans1b_max],axis=3)

    remix_weights = self.weight_variable(name+"/w1r",[global_mid_channels*2,mid_channels],global_mid_channels*2,mid_channels, scale_initial_weights = 0.5)
    conv1_layer = conv1a_layer + tf.tensordot(trans1b_pooled,remix_weights,axes=[[3],[0]])

    trans2_layer = self.parametric_relu(name+"/prelu2",(self.batchnorm(name+"/norm2",conv1_layer)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels, main_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer

  #Convolutional residual block with internal batch norm and nonlinear activation
  def dilated_res_conv_block(self, name, in_layer, diam, main_channels, mid_channels, dilated_mid_channels, dilation, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None):
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    weights1a = self.conv_weight_variable(name+"/w1a", diam, diam, main_channels, mid_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    weights1b = self.conv_weight_variable(name+"/w1b", diam, diam, main_channels, dilated_mid_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    conv1a_layer = self.conv2d(trans1_layer, weights1a)
    conv1b_layer = self.dilated_conv2d(trans1_layer, weights1b, dilation=dilation)
    self.outputs_by_layer.append((name+"/conv1a",conv1a_layer))
    self.outputs_by_layer.append((name+"/conv1b",conv1b_layer))

    conv1_layer = tf.concat([conv1a_layer,conv1b_layer],axis=3)

    trans2_layer = self.parametric_relu(name+"/prelu2",(self.batchnorm(name+"/norm2",conv1_layer)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels+dilated_mid_channels, main_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer

  #Convolutional residual block that does sequential horizontal and vertical convolutions, with internal batch norm and nonlinear activation
  def hv_res_conv_block(self, name, in_layer, diam, main_channels, mid_channels):
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    weights1 = self.weight_variable(name+"/w1",[diam,1,main_channels,mid_channels],main_channels*diam,mid_channels)
    weights2 = self.weight_variable(name+"/w2",[1,diam,mid_channels,main_channels],main_channels*diam,mid_channels)

    conv1_layer = self.conv2d(trans1_layer, weights1)
    self.outputs_by_layer.append((name+"/conv1",conv1_layer))

    trans2_layer = self.parametric_relu(name+"/prelu2",(self.batchnorm(name+"/norm2",conv1_layer)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer * 0.5

  #Same, but vertical then horizontal
  def vh_res_conv_block(self, name, in_layer, diam, main_channels, mid_channels):
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    weights1 = self.weight_variable(name+"/w1",[1,diam,main_channels,mid_channels],main_channels*diam,mid_channels)
    weights2 = self.weight_variable(name+"/w2",[diam,1,mid_channels,main_channels],main_channels*diam,mid_channels)

    conv1_layer = self.conv2d(trans1_layer, weights1)
    self.outputs_by_layer.append((name+"/conv1",conv1_layer))

    trans2_layer = self.parametric_relu(name+"/prelu2",(self.batchnorm(name+"/norm2",conv1_layer)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer * 0.5


  def chainpool_block(self, name, in_layer, chains, num_chain_segments, empty, nonempty, diam, main_channels, mid_channels):
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    weights1max = self.conv_weight_variable(name+"/w1max", diam, diam, main_channels, mid_channels)
    # weights1sum = self.conv_weight_variable(name+"/w1sum", diam, diam, main_channels, mid_channels)
    conv1max_layer = self.conv2d(trans1_layer, weights1max)
    # conv1sum_layer = self.conv2d(trans1_layer, weights1sum)
    self.outputs_by_layer.append((name+"/conv1max",conv1max_layer))
    # self.outputs_by_layer.append((name+"/conv1sum",conv1sum_layer))

    trans2max_layer = self.parametric_relu(name+"/prelu2max",(self.batchnorm(name+"/norm2max",conv1max_layer)))
    # trans2sum_layer = self.parametric_relu(name+"/prelu2sum",(self.batchnorm(name+"/norm2sum",conv1sum_layer)))
    self.outputs_by_layer.append((name+"/trans2max",trans2max_layer))
    # self.outputs_by_layer.append((name+"/trans2sum",trans2sum_layer))

    maxpooled_layer = self.chain_pool(trans2max_layer,chains,num_chain_segments,empty,nonempty,mode="max")
    # sumpooled_layer = self.chain_pool(trans2sum_layer,chains,empty,nonempty,mode="sum")
    self.outputs_by_layer.append((name+"/maxpooled",maxpooled_layer))
    # self.outputs_by_layer.append((name+"/sumpooled",sumpooled_layer))

    pooled_layer = maxpooled_layer
    #pooled_layer = tf.concat([maxpooled_layer,sumpooled_layer],axis=3)

    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels, main_channels)
    conv2_layer = self.conv2d(pooled_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer


  #Special block for detecting ladders, with mid_channels channels per each of 4 diagonal scans.
  def ladder_block(self, name, in_layer, near_nonempty, main_channels, mid_channels):
    # Converts [[123][456][789]] to [[12300][04560][00789]]
    def skew_right(tensor):
      n = self.max_board_size
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
      n = self.max_board_size
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
      n = self.max_board_size
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
      n = self.max_board_size
      assert(tensor.shape[1].value == n)
      assert(tensor.shape[2].value == 2*n-1)
      c = tensor.shape[3].value
      tensor = tf.reshape(tensor,[-1,n*(2*n-1),c]) #Linearize
      tensor = tf.pad(tensor,[[0,0],[2*n-3,n-1],[0,0]]) #Pad 19*37 -> 21*36
      tensor = tf.reshape(tensor,[-1,n+2,2*n-2,c]) #Convert back to 21x36
      tensor = tensor[:,1:(n+1),(n-2):,:] #Chop off the extra, now we are 19x19
      return tensor

    #First, as usual, batchnorm and relu the trunk to get the values to a reasonable scale
    trans1_layer = self.parametric_relu(name+"/prelu1",(self.batchnorm(name+"/norm1",in_layer)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    c = mid_channels

    #The next part basically does a scan across the board each of the 4 diagonal ways, computing a moving average.
    #We use a convolution to let the neural net choose the values and weights:
    #a: value on this spot to be moving-averaged
    #b: if the weight on the moving average so far is 1, the value on this spot gets a factor of exp(b)-1 weight.
    diampre = 3
    weightsprea = self.conv_weight_variable(name+"/wprea", diampre, diampre, main_channels, c*4)
    weightspreb = self.conv_weight_variable(name+"/wpreb", diampre, diampre, main_channels, c*4)

    convprea_layer = self.conv2d(trans1_layer, weightsprea)
    convpreb_layer = self.conv2d(trans1_layer, weightspreb)
    self.outputs_by_layer.append((name+"/convprea",convprea_layer))
    self.outputs_by_layer.append((name+"/convpreb",convpreb_layer))

    assert(len(near_nonempty.shape) == 4)
    assert(near_nonempty.shape[1].value == self.max_board_size)
    assert(near_nonempty.shape[2].value == self.max_board_size)
    assert(near_nonempty.shape[3].value == 1)

    transprea_layer = self.parametric_relu(name+"/preluprea",(self.batchnorm(name+"/normprea",convprea_layer)))
    transpreb_layer = tf.nn.sigmoid(self.batchnorm(name+"/normpreb",convpreb_layer)) * near_nonempty * 1.5 + 0.0001
    self.outputs_by_layer.append((name+"/transprea",transprea_layer))
    self.outputs_by_layer.append((name+"/transpreb",transpreb_layer))

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
    weightspost = self.conv_weight_variable(name+"/wpost", diampost, diampost, c*4, main_channels)

    convpost_layer = self.conv2d(results, weightspost)
    self.outputs_by_layer.append((name+"/convpost",convpost_layer))

    return convpost_layer


  #Begin Neural net------------------------------------------------------------------------------------
  #Indexing:
  #batch, bsize, bsize, channel

  def build_model(self, use_ranks, include_policy, include_value, predict_pass):
    max_board_size = self.max_board_size

    #Input layer---------------------------------------------------------------------------------
    inputs = tf.placeholder(tf.float32, [None] + self.input_shape)
    ranks = tf.placeholder(tf.float32, [None] + self.rank_shape)
    symmetries = tf.placeholder(tf.bool, [3])
    self.inputs = inputs
    self.ranks = ranks
    self.symmetries = symmetries

    features_active = tf.constant([
      1.0, #0
      1.0, #1
      1.0, #2
      1.0, #3
      1.0, #4
      1.0, #5
      1.0, #6
      1.0, #7
      1.0, #8
      1.0, #9
      1.0, #10
      1.0, #11
      1.0, #12
      1.0, #13
      1.0, #14
      1.0, #15
      1.0, #16
      1.0, #17
      1.0, #18
    ])
    assert(features_active.dtype == tf.float32)

    cur_layer = tf.reshape(inputs, [-1] + self.post_input_shape)

    input_num_channels = self.post_input_shape[2]
    #Input symmetries - we apply symmetries during training by transforming the input and reverse-transforming the output
    cur_layer = self.apply_symmetry(cur_layer,symmetries,inverse=False)
    #Disable various features
    cur_layer = cur_layer * tf.reshape(features_active,[1,1,1,-1])

    # nonempty = cur_layer[:,:,:,1] + cur_layer[:,:,:,2]
    # empty = 1.0 - nonempty
    # near_nonempty = tf.minimum(1.0,self.conv2d(tf.expand_dims(nonempty,axis=3),manhattan_radius_3_kernel))

    #Transform and append ranks
    if use_ranks:
      rank_embedding_weights = self.weight_variable("rankembedding/w",[self.rank_shape[0],self.rank_embedding_dim],self.rank_shape[0],self.rank_embedding_dim)
      rank_embedding_layer = tf.tensordot(ranks,rank_embedding_weights,axes=[[1],[0]])
      rank_embedding_layer = tf.tile(tf.reshape(rank_embedding_layer, [-1,1,1,self.rank_embedding_dim]), [1,max_board_size,max_board_size,1])
      cur_layer = tf.concat([cur_layer,rank_embedding_layer], axis=3)
      input_num_channels += self.rank_embedding_dim

    #Convolutional RELU layer 1-------------------------------------------------------------------------------------
    trunk = self.conv_only_extra_center_block("conv1",cur_layer,diam=5,in_channels=input_num_channels,out_channels=224)

    #Residual Convolutional Block 1---------------------------------------------------------------------------------
    residual = self.res_conv_block("rconv1",trunk,diam=3,main_channels=224,mid_channels=224, emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv1",trunk,residual)

    #Residual Convolutional Block 2---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv2",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv2",trunk,residual)

    #Residual Convolutional Block 3---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv3",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv3",trunk,residual)

    #Residual Convolutional Block 4---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv4",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv4",trunk,residual)

    #Residual Convolutional Block 5---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv5",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv5",trunk,residual)

    # #Residual Convolutional Block 6---------------------------------------------------------------------------------
    # residual = self.dilated_res_conv_block("rconv6",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
    #                                        emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    # trunk = self.merge_residual("rconv6",trunk,residual)

    #Residual Convolutional Block 7---------------------------------------------------------------------------------
    residual = self.global_res_conv_block("rconv7",trunk,diam=3,main_channels=224,mid_channels=160, global_mid_channels=64,
                                          emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv7",trunk,residual)

    #Residual Convolutional Block 8---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv8",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv8",trunk,residual)

    #Residual Convolutional Block 9---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv9",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv9",trunk,residual)

    #Residual Convolutional Block 10---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv10",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv10",trunk,residual)

    #Residual Convolutional Block 11---------------------------------------------------------------------------------
    residual = self.global_res_conv_block("rconv11",trunk,diam=3,main_channels=224,mid_channels=160, global_mid_channels=64,
                                          emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv11",trunk,residual)

    # #Residual Convolutional Block 12---------------------------------------------------------------------------------
    # residual = self.dilated_res_conv_block("rconv12",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
    #                                        emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    # trunk = self.merge_residual("rconv12",trunk,residual)

    #Residual Convolutional Block 13---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv13",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv13",trunk,residual)

    #Residual Convolutional Block 14---------------------------------------------------------------------------------
    residual = self.dilated_res_conv_block("rconv14",trunk,diam=3,main_channels=224,mid_channels=160, dilated_mid_channels=64, dilation=2,
                                           emphasize_center_weight = 0.3, emphasize_center_lr=1.5)
    trunk = self.merge_residual("rconv14",trunk,residual)

    #Postprocessing residual trunk----------------------------------------------------------------------------------

    #Normalize and relu just before the policy head
    trunk = self.parametric_relu("trunk/prelu",(self.batchnorm("trunk/norm",trunk)))
    self.outputs_by_layer.append(("trunk",trunk))


    #Policy head---------------------------------------------------------------------------------
    if include_policy:
      p0_layer = trunk

      #This is the main path for policy information
      p1_num_channels = 48
      p1_intermediate_conv = self.conv_only_block("p1/intermediate_conv",p0_layer,diam=3,in_channels=224,out_channels=p1_num_channels)

      #But in parallel convolve to compute some features about the global state of the board
      #Hopefully the neural net uses this for stuff like ko situation, overall temperature/threatyness, who is leading, etc.
      g1_num_channels = 32
      g1_layer = self.conv_block("g1",p0_layer,diam=3,in_channels=224,out_channels=g1_num_channels)

      #Fold g1 down to single values for the board.
      #For stdev, add a tiny constant to ensure numeric stability
      g1_mean = tf.reduce_mean(g1_layer,axis=[1,2],keepdims=True)
      g1_max = tf.reduce_max(g1_layer,axis=[1,2],keepdims=True)
      g2_layer = tf.concat([g1_mean,g1_max],axis=3) #shape [b,1,1,2*convg1num_channels]
      g2_num_channels = 2*g1_num_channels
      self.outputs_by_layer.append(("g2",g2_layer))

      #Transform them into the space of the policy features to act as biases for the policy
      #Also divide the initial weights a bit more because we think these should matter a bit less than local shape stuff,
      #by multiplying the number of inputs for purposes of weight initialization (currently mult by 4)
      matmulg2w = self.weight_variable("matmulg2w",[g2_num_channels,p1_num_channels],g2_num_channels*4,p1_num_channels)
      g3_layer = tf.tensordot(g2_layer,matmulg2w,axes=[[3],[0]])
      self.outputs_by_layer.append(("g3",g3_layer))

      #Add! This adds shapes [b,19,19,convp1_num_channels] + [b,1,1,convp1_num_channels]
      #so the second one should get broadcast up to the size of the first one.
      #We can think of p1 as being an ordinary convolution layer except that for every node of the convolution, the g2 values (g2_num_channels many of them)
      #have been appended to the p0 incoming values (p0_num_channels * convp1diam * convp1diam many of them).
      #The matrix matmulg2w is simply the set of weights for that additional part of the matrix. It's just that rather than appending beforehand,
      #we multiply separately and add to the output afterward.
      p1_intermediate_sum = p1_intermediate_conv + g3_layer

      #And now apply batchnorm and crelu
      p1_layer = tf.nn.crelu(self.batchnorm("p1/norm",p1_intermediate_sum))
      self.outputs_by_layer.append(("p1",p1_layer))

      #Finally, apply linear convolution to produce final output
      #2x in_channels due to crelu
      p2_layer = self.conv_only_block("p2",p1_layer,diam=5,in_channels=p1_num_channels*2,out_channels=1,scale_initial_weights=0.5,reg=False)

      self.add_lr_factor("p1/norm/beta:0",0.25)
      self.add_lr_factor("p2/w:0",0.25)

      #Output symmetries - we apply symmetries during training by transforming the input and reverse-transforming the output
      policy_output = self.apply_symmetry(p2_layer,symmetries,inverse=True)
      policy_output = tf.reshape(policy_output, [-1] + self.policy_target_shape_nopass)

      if not predict_pass:
        #Simply add the pass output on with a large negative constant that's probably way more negative than anything
        #else the neural net would output.
        policy_output = tf.pad(policy_output,[(0,0),(0,1)], constant_values = -10000.)
      else:
        #Add pass move based on the global g values
        matmulpass = self.weight_variable("matmulpass",[g2_num_channels,1],g2_num_channels*8,1)
        self.add_lr_factor("matmulpass:0",0.25)
        pass_output = tf.tensordot(g2_layer,matmulpass,axes=[[3],[0]])
        self.outputs_by_layer.append(("pass",pass_output))
        pass_output = tf.reshape(pass_output, [-1] + [1])
        policy_output = tf.concat([policy_output,pass_output],axis=1)

      self.policy_output = policy_output
    else:
      #Don't include policy? Just set the policy output to all zeros.
      policy_output = tf.zeros_like(inputs[:,:,0])
      policy_output = tf.pad(policy_output,[(0,0),(0,1)])
      self.policy_output = policy_output

    if include_value:
      v0_layer = trunk

      v1_num_channels = 8
      v1_layer = self.conv_block("v1",v0_layer,diam=3,in_channels=224,out_channels=v1_num_channels)
      self.outputs_by_layer.append(("v1",v1_layer))

      v1_layer_pooled = tf.reduce_mean(v1_layer,axis=[1,2],keepdims=False)
      v1_size = v1_num_channels

      v2_size = 8
      v2w = self.weight_variable("v2/w",[v1_size,v2_size],v1_size,v2_size)
      v2b = self.weight_variable("v2/b",[v2_size],v1_size,v2_size,scale_initial_weights=0.2,reg=False)
      v2_layer = tf.nn.crelu(tf.matmul(v1_layer_pooled, v2w) + v2b)
      v2_size *= 2 #for crelu

      v3_size = 1
      v3w = self.weight_variable("v3/w",[v2_size,v3_size],v2_size,v3_size)
      v3b = self.weight_variable("v3/b",[v3_size],v2_size,v3_size,scale_initial_weights=0.2,reg=False)
      v3_layer = tf.matmul(v2_layer, v3w) + v3b

      value_output = tf.reshape(v3_layer, [-1] + self.value_target_shape)

      self.value_output = value_output
    else:
      self.value_output = tf.zeros_like(inputs[:,0,0])



class Target_vars:
  def __init__(self,model,for_optimization,require_last_move):
    policy_output = model.policy_output
    value_output = model.value_output

    #Loss function
    self.policy_targets = tf.placeholder(tf.float32, [None] + model.policy_target_shape)
    self.value_target = tf.placeholder(tf.float32, [None] + model.value_target_shape)
    self.target_weights_from_data = tf.placeholder(tf.float32, [None] + model.target_weights_shape)

    if require_last_move == "all":
      self.target_weights_used = self.target_weights_from_data * tf.reduce_sum(model.inputs[:,:,14],axis=[1])
    elif require_last_move is True:
      self.target_weights_used = self.target_weights_from_data * tf.reduce_sum(model.inputs[:,:,10],axis=[1])
    else:
      self.target_weights_used = self.target_weights_from_data

    self.policy_loss = tf.reduce_sum(
      self.target_weights_used *
      tf.nn.softmax_cross_entropy_with_logits(labels=self.policy_targets, logits=policy_output)
    )

    cross_entropy_value_loss = 1.2*tf.reduce_sum(
      self.target_weights_used *
      tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stack([(1+self.value_target)/2,(1-self.value_target)/2],axis=1),
        logits=tf.stack([value_output,tf.zeros_like(value_output)],axis=1)
      )
    )

    l2_value_loss = tf.reduce_sum(
      self.target_weights_used *
      tf.square(self.value_target - tf.tanh(value_output))
    )

    self.value_loss = 0.5 * (cross_entropy_value_loss + l2_value_loss)

    if for_optimization:
      #Prior/Regularization
      self.l2_reg_coeff = tf.placeholder(tf.float32)
      self.weight_sum = tf.reduce_sum(self.target_weights_used)
      self.reg_loss = self.l2_reg_coeff * tf.add_n([tf.nn.l2_loss(variable) for variable in model.reg_variables]) * self.weight_sum

      #The loss to optimize
      self.opt_loss = self.policy_loss + self.value_loss + self.reg_loss

class Metrics:
  def __init__(self,model,target_vars,include_debug_stats):
    #Training results
    policy_target_idxs = tf.argmax(target_vars.policy_targets, 1)
    self.top1_prediction = tf.equal(tf.argmax(model.policy_output, 1), policy_target_idxs)
    self.top4_prediction = tf.nn.in_top_k(model.policy_output,policy_target_idxs,4)
    self.accuracy1 = tf.reduce_sum(target_vars.target_weights_used * tf.cast(self.top1_prediction, tf.float32))
    self.accuracy4 = tf.reduce_sum(target_vars.target_weights_used * tf.cast(self.top4_prediction, tf.float32))

    #Debugging stats
    if include_debug_stats:

      def reduce_norm(x, axis=None, keepdims=False):
        return tf.sqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=keepdims))

      def reduce_stdev(x, axis=None, keepdims=False):
        m = tf.reduce_mean(x, axis=axis, keepdims=True)
        devs_squared = tf.square(x - m)
        return tf.sqrt(tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims))

      self.activated_prop_by_layer = dict([
        (name,tf.reduce_mean(tf.count_nonzero(layer,axis=[1,2])/layer.shape[1].value/layer.shape[2].value, axis=0)) for (name,layer) in model.outputs_by_layer
      ])
      self.mean_output_by_layer = dict([
        (name,tf.reduce_mean(layer,axis=[0,1,2])) for (name,layer) in model.outputs_by_layer
      ])
      self.stdev_output_by_layer = dict([
        (name,reduce_stdev(layer,axis=[0,1,2])**2) for (name,layer) in model.outputs_by_layer
      ])
      self.mean_weights_by_var = dict([
        (v.name,tf.reduce_mean(v)) for v in tf.trainable_variables()
      ])
      self.norm_weights_by_var = dict([
        (v.name,reduce_norm(v)) for v in tf.trainable_variables()
      ])
