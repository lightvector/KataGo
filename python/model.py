import logging
import math
import traceback
import tensorflow as tf
import numpy as np

from board import Board

#Feature extraction functions-------------------------------------------------------------------

class Model:
  # Sizes of the various array targets in the data, used also to size some of the output head dimensions
  # for auxiliary outputs
  NUM_POLICY_TARGETS = 2
  NUM_GLOBAL_TARGETS = 64
  NUM_VALUE_SPATIAL_TARGETS = 5
  EXTRA_SCORE_DISTR_RADIUS = 60

  @staticmethod
  def get_version(config):
    if "version" in config:
      return config["version"]
    return 5 #by default, since this was the version before we put it in the config

  @staticmethod
  def get_num_bin_input_features(config):
    version = Model.get_version(config)
    if version == 4:
      return 22
    elif version == 5:
      return 22
    elif version == 6:
      return 13
    elif version == 7:
      return 22
    elif version == 8:
      return 22
    else:
      assert(False)

  @staticmethod
  def get_num_global_input_features(config):
    version = Model.get_version(config)
    if version == 4:
      return 14
    elif version == 5:
      return 14
    elif version == 6:
      return 12
    elif version == 7:
      return 16
    elif version == 8:
      return 19
    else:
      assert(False)


  def __init__(self,config,pos_len,placeholders,is_training=False):
    self.pos_len = pos_len
    self.num_bin_input_features = Model.get_num_bin_input_features(config)
    self.num_global_input_features = Model.get_num_global_input_features(config)

    self.bin_input_shape = [self.pos_len*self.pos_len,self.num_bin_input_features]
    self.binp_input_shape = [self.num_bin_input_features,(self.pos_len*self.pos_len+7)//8]
    self.global_input_shape = [self.num_global_input_features]
    self.post_input_shape = [self.pos_len,self.pos_len,self.num_bin_input_features]
    self.policy_output_shape_nopass = [self.pos_len*self.pos_len,2]
    self.policy_output_shape = [self.pos_len*self.pos_len+1,2] #+1 for pass move
    self.policy_target_shape = [self.pos_len*self.pos_len+1] #+1 for pass move
    self.policy_target_weight_shape = []
    self.value_target_shape = [3]
    self.td_value_target_shape = [2,3]
    self.miscvalues_target_shape = [10] #0:scoremean, #1 scorestdev, #2 lead, #3 variance time #4-#9 td value targets
    self.scoremean_target_shape = [] #0
    self.scorestdev_target_shape = [] #1
    self.lead_target_shape = [] #2
    self.variance_time_target_shape = [] #3
    self.scorebelief_target_shape = [self.pos_len*self.pos_len*2+Model.EXTRA_SCORE_DISTR_RADIUS*2]
    self.ownership_target_shape = [self.pos_len,self.pos_len]
    self.scoring_target_shape = [self.pos_len,self.pos_len]
    self.futurepos_target_shape = [self.pos_len,self.pos_len,2]
    self.seki_output_shape = [self.pos_len,self.pos_len,4]
    self.seki_target_shape = [self.pos_len,self.pos_len]
    self.target_weight_shape = []
    self.lead_target_weight_shape = []
    self.ownership_target_weight_shape = []
    self.scoring_target_weight_shape = []
    self.futurepos_target_weight_shape = []

    self.pass_pos = self.pos_len * self.pos_len

    self.reg_variables = []
    self.reg_variables_tiny = []
    self.prescale_variables = []
    self.lr_adjusted_variables = {}
    self.is_training = is_training
    self.is_training_tensor = tf.constant(is_training,dtype=tf.bool)

    self.num_blocks = len(config["block_kind"])
    self.use_fixup = config["use_fixup"]

    #Accumulates outputs for printing stats about their activations
    self.outputs_by_layer = []
    self.other_internal_outputs = []
    #Accumulates info about batch norm laywers
    self.batch_norms = {}

    self.support_japanese_rules = True #by default
    if "support_japanese_rules" in config and config["support_japanese_rules"] == False:
      self.support_japanese_rules = False

    self.build_model(config,placeholders)

  def assert_batched_shape(self,name,tensor,shape):
    if (len(tensor.shape) != len(shape)+1 or
        [int(tensor.shape[i+1].value) for i in range(len(shape))] != [int(x) for x in shape]):
      raise Exception("%s should have shape %s after a batch dimension but instead it had shape %s" % (
        name, str(shape), str([str(x.value) for x in tensor.shape])))

  def assert_shape(self,name,tensor,shape):
    if (len(tensor.shape) != len(shape) or
        [int(x.value) for x in tensor.shape] != [int(x) for x in shape]):
      raise Exception("%s should have shape %s but instead it had shape %s" % (
        name, str(shape), str([str(x.value) for x in tensor.shape])))

  def xy_to_tensor_pos(self,x,y):
    return y * self.pos_len + x
  def loc_to_tensor_pos(self,loc,board):
    assert(loc != Board.PASS_LOC)
    return board.loc_y(loc) * self.pos_len + board.loc_x(loc)

  def tensor_pos_to_loc(self,pos,board):
    if pos == self.pass_pos:
      return None
    pos_len = self.pos_len
    bsize = board.size
    assert(self.pos_len >= bsize)
    x = pos % pos_len
    y = pos // pos_len
    if x < 0 or x >= bsize or y < 0 or y >= bsize:
      return board.loc(-10,-10) #Return an illegal move since this is offboard
    return board.loc(x,y)

  def sym_tensor_pos(self,pos,symmetry):
    if pos == self.pass_pos:
      return pos
    pos_len = self.pos_len
    x = pos % pos_len
    y = pos // pos_len
    if symmetry >= 4:
      symmetry -= 4
      tmp = x
      x = y
      y = tmp
    if symmetry >= 2:
      symmetry -= 2
      x = pos_len-x-1
    if symmetry >= 1:
      symmetry -= 1
      y = pos_len-y-1
    return y * pos_len + x

  #Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
  def iterLadders(self, board, f):
    chainHeadsSolved = {}
    copy = board.copy()

    bsize = board.size
    assert(self.pos_len >= bsize)

    for y in range(bsize):
      for x in range(bsize):
        pos = self.xy_to_tensor_pos(x,y)
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
  def fill_row_features(self, board, pla, opp, boards, moves, move_idx, rules, bin_input_data, global_input_data, idx):
    #Currently only support v4 or v5 or v7 MODEL features (inputs version v3 and v4 and v6)
    assert(self.version == 4 or self.version == 5 or self.version == 7 or self.version == 8)

    bsize = board.size
    assert(self.pos_len >= bsize)
    assert(len(boards) > 0)
    assert(board.zobrist == boards[move_idx].zobrist)

    for y in range(bsize):
      for x in range(bsize):
        pos = self.xy_to_tensor_pos(x,y)
        bin_input_data[idx,pos,0] = 1.0
        loc = board.loc(x,y)
        stone = board.board[loc]
        if stone == pla:
          bin_input_data[idx,pos,1] = 1.0
        elif stone == opp:
          bin_input_data[idx,pos,2] = 1.0

        if stone == pla or stone == opp:
          libs = board.num_liberties(loc)
          if libs == 1:
            bin_input_data[idx,pos,3] = 1.0
          elif libs == 2:
            bin_input_data[idx,pos,4] = 1.0
          elif libs == 3:
            bin_input_data[idx,pos,5] = 1.0

    #Python code does NOT handle superko
    if board.simple_ko_point is not None:
      pos = self.loc_to_tensor_pos(board.simple_ko_point,board)
      bin_input_data[idx,pos,6] = 1.0
    #Python code does NOT handle ko-prohibited encore spots or anything relating to the encore
    #so features 7 and 8 leave that blank

    if move_idx >= 1 and moves[move_idx-1][0] == opp:
      prev1_loc = moves[move_idx-1][1]
      if prev1_loc is not None and prev1_loc != Board.PASS_LOC:
        pos = self.loc_to_tensor_pos(prev1_loc,board)
        bin_input_data[idx,pos,9] = 1.0
      elif prev1_loc == Board.PASS_LOC:
        global_input_data[idx,0] = 1.0

      if move_idx >= 2 and moves[move_idx-2][0] == pla:
        prev2_loc = moves[move_idx-2][1]
        if prev2_loc is not None and prev2_loc != Board.PASS_LOC:
          pos = self.loc_to_tensor_pos(prev2_loc,board)
          bin_input_data[idx,pos,10] = 1.0
        elif prev2_loc == Board.PASS_LOC:
          global_input_data[idx,1] = 1.0

        if move_idx >= 3 and moves[move_idx-3][0] == opp:
          prev3_loc = moves[move_idx-3][1]
          if prev3_loc is not None and prev3_loc != Board.PASS_LOC:
            pos = self.loc_to_tensor_pos(prev3_loc,board)
            bin_input_data[idx,pos,11] = 1.0
          elif prev3_loc == Board.PASS_LOC:
            global_input_data[idx,2] = 1.0

          if move_idx >= 4 and moves[move_idx-4][0] == pla:
            prev4_loc = moves[move_idx-4][1]
            if prev4_loc is not None and prev4_loc != Board.PASS_LOC:
              pos = self.loc_to_tensor_pos(prev4_loc,board)
              bin_input_data[idx,pos,12] = 1.0
            elif prev4_loc == Board.PASS_LOC:
              global_input_data[idx,3] = 1.0

            if move_idx >= 5 and moves[move_idx-5][0] == opp:
              prev5_loc = moves[move_idx-5][1]
              if prev5_loc is not None and prev5_loc != Board.PASS_LOC:
                pos = self.loc_to_tensor_pos(prev5_loc,board)
                bin_input_data[idx,pos,13] = 1.0
              elif prev5_loc == Board.PASS_LOC:
                global_input_data[idx,4] = 1.0

    def addLadderFeature(loc,pos,workingMoves):
      assert(board.board[loc] == Board.BLACK or board.board[loc] == Board.WHITE)
      bin_input_data[idx,pos,14] = 1.0
      if board.board[loc] == opp and board.num_liberties(loc) > 1:
        for workingMove in workingMoves:
          workingPos = self.loc_to_tensor_pos(workingMove,board)
          bin_input_data[idx,workingPos,17] = 1.0

    self.iterLadders(board, addLadderFeature)

    if move_idx > 0:
      prevBoard = boards[move_idx-1]
    else:
      prevBoard = board
    def addPrevLadderFeature(loc,pos,workingMoves):
      assert(prevBoard.board[loc] == Board.BLACK or prevBoard.board[loc] == Board.WHITE)
      bin_input_data[idx,pos,15] = 1.0
    self.iterLadders(prevBoard, addPrevLadderFeature)

    if move_idx > 1:
      prevPrevBoard = boards[move_idx-2]
    else:
      prevPrevBoard = prevBoard
    def addPrevPrevLadderFeature(loc,pos,workingMoves):
      assert(prevPrevBoard.board[loc] == Board.BLACK or prevPrevBoard.board[loc] == Board.WHITE)
      bin_input_data[idx,pos,16] = 1.0
    self.iterLadders(prevPrevBoard, addPrevPrevLadderFeature)

    #Features 18,19 - area
    area = [0 for i in range(board.arrsize)]

    if self.version <= 6:
      if rules["scoringRule"] == "SCORING_AREA":
        nonPassAliveStones = False
        safeBigTerritories = True
        unsafeBigTerritories = False
        if self.version != 5:
          nonPassAliveStones = True
          unsafeBigTerritories = True
        board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])
      elif rules["scoringRule"] == "SCORING_TERRITORY":
        nonPassAliveStones = False
        safeBigTerritories = True
        unsafeBigTerritories = False
        board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])
      else:
        assert(False)
    else:
      if rules["scoringRule"] == "SCORING_AREA" and rules["taxRule"] == "TAX_NONE":
        safeBigTerritories = True
        nonPassAliveStones = True
        unsafeBigTerritories = True
        board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])
      else:
        hasAreaFeature = False
        keepTerritories = False
        keepStones = False
        if rules["scoringRule"] == "SCORING_AREA" and (rules["taxRule"] == "TAX_SEKI" or rules["taxRule"] == "TAX_ALL"):
          hasAreaFeature = True
          keepTerritories = False
          keepStones = True
        elif rules["scoringRule"] == "SCORING_TERRITORY" and rules["taxRule"] == "TAX_NONE":
          if rules["encorePhase"] >= 2:
            hasAreaFeature = True
            keepTerritories = True
            keepStones = False
        elif rules["scoringRule"] == "SCORING_TERRITORY" and (rules["taxRule"] == "TAX_SEKI" or rules["taxRule"] == "TAX_ALL"):
          if rules["encorePhase"] >= 2:
            hasAreaFeature = True
            keepTerritories = False
            keepStones = False
        else:
          assert(False)

        if hasAreaFeature:
          board.calculateNonDameTouchingArea(
            area,
            keepTerritories,
            keepStones,
            rules["multiStoneSuicideLegal"]
          )

    for y in range(bsize):
      for x in range(bsize):
        loc = board.loc(x,y)
        pos = self.xy_to_tensor_pos(x,y)

        if area[loc] == pla:
          bin_input_data[idx,pos,18] = 1.0
        elif area[loc] == opp:
          bin_input_data[idx,pos,19] = 1.0

    #Features 20,21 - second encore phase starting stones, we just set them to the current stones in pythong
    #since we don't really have a jp rules impl
    if rules["encorePhase"] >= 2:
      for y in range(bsize):
        for x in range(bsize):
          pos = self.xy_to_tensor_pos(x,y)
          loc = board.loc(x,y)
          stone = board.board[loc]
          if stone == pla:
            bin_input_data[idx,pos,20] = 1.0
          elif stone == opp:
            bin_input_data[idx,pos,21] = 1.0


    #Not quite right, japanese rules aren't really implemented in the python
    bArea = board.size * board.size
    whiteKomi = rules["whiteKomi"]
    if rules["scoringRule"] == "SCORING_TERRITORY":
      selfKomi = (whiteKomi if pla == Board.WHITE else -whiteKomi+1)
    else:
      selfKomi = (whiteKomi if pla == Board.WHITE else -whiteKomi)

    if selfKomi > bArea+1:
      selfKomi = bArea+1
    if selfKomi < -bArea-1:
      selfKomi = -bArea-1
    if self.version >= 7:
      global_input_data[idx,5] = selfKomi/20.0
    else:
      global_input_data[idx,5] = selfKomi/15.0

    if rules["koRule"] == "KO_SIMPLE":
      pass
    elif rules["koRule"] == "KO_POSITIONAL" or rules["koRule"] == "KO_SPIGHT":
      global_input_data[idx,6] = 1.0
      global_input_data[idx,7] = 0.5
    elif rules["koRule"] == "KO_SITUATIONAL":
      global_input_data[idx,6] = 1.0
      global_input_data[idx,7] = -0.5
    else:
      assert(False)

    if rules["multiStoneSuicideLegal"]:
      global_input_data[idx,8] = 1.0

    if rules["scoringRule"] == "SCORING_AREA":
      pass
    elif rules["scoringRule"] == "SCORING_TERRITORY":
      global_input_data[idx,9] = 1.0
    else:
      assert(False)

    if self.version >= 7:
      if rules["taxRule"] == "TAX_NONE":
        pass
      elif rules["taxRule"] == "TAX_SEKI":
        global_input_data[idx,10] = 1.0
      elif rules["taxRule"] == "TAX_ALL":
        global_input_data[idx,10] = 1.0
        global_input_data[idx,11] = 1.0
      else:
        assert(False)

    if self.version >= 7:
      if rules["encorePhase"] > 0:
        global_input_data[idx,12] = 1.0
      if rules["encorePhase"] > 1:
        global_input_data[idx,13] = 1.0
      passWouldEndPhase = rules["passWouldEndPhase"]
      global_input_data[idx,14] = (1.0 if passWouldEndPhase else 0.0)
    else:
      if rules["encorePhase"] > 0:
        global_input_data[idx,10] = 1.0
      if rules["encorePhase"] > 1:
        global_input_data[idx,11] = 1.0
      passWouldEndPhase = rules["passWouldEndPhase"]
      global_input_data[idx,12] = (1.0 if passWouldEndPhase else 0.0)

    if self.version >= 8 and "asymPowersOfTwo" in rules:
      global_input_data[idx,15] = 1.0
      global_input_data[idx,16] = rules["asymPowersOfTwo"]

    if self.version >= 8:
      if "hasButton" in rules and rules["hasButton"] and Board.PASS_LOC not in [move[1] for move in moves]:
        global_input_data[idx,17] = 1.0

    if rules["scoringRule"] == "SCORING_AREA" or rules["encorePhase"] > 1:
      boardAreaIsEven = (board.size % 2 == 0)

      drawableKomisAreEven = boardAreaIsEven

      if drawableKomisAreEven:
        komiFloor = math.floor(selfKomi / 2.0) * 2.0
      else:
        komiFloor = math.floor((selfKomi-1.0) / 2.0) * 2.0 + 1.0

      delta = selfKomi - komiFloor
      assert(delta >= -0.0001)
      assert(delta <= 2.0001)
      if delta < 0.0:
        delta = 0.0
      if delta > 2.0:
        delta = 2.0

      if delta < 0.5:
        wave = delta
      elif delta < 1.5:
        wave = 1.0-delta
      else:
        wave = delta-2.0

      if self.version >= 8:
        global_input_data[idx,18] = wave
      elif self.version >= 7:
        global_input_data[idx,15] = wave
      else:
        global_input_data[idx,13] = wave

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

  def batchnorm_and_mask(self,name,tensor,mask,mask_sum,use_gamma_in_fixup=False):
    if self.use_fixup:
      self.batch_norms[name] = (tensor.shape[-1].value,1e-20,True,use_gamma_in_fixup,self.use_fixup)
      if use_gamma_in_fixup:
        gamma = self.weight_variable_init_constant(name+"/gamma", [tensor.shape[3].value], 1.0)
        beta = self.weight_variable_init_constant(name+"/beta", [tensor.shape[3].value], 0.0, reg="tiny")
        return (tensor * gamma + beta) * mask
      else:
        beta = self.weight_variable_init_constant(name+"/beta", [tensor.shape[3].value], 0.0, reg="tiny")
        return (tensor + beta) * mask

    epsilon = 0.001
    has_bias = True
    has_scale = False
    self.batch_norms[name] = (tensor.shape[-1].value,epsilon,has_bias,has_scale,self.use_fixup)

    num_channels = tensor.shape[3].value
    collections = [tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.MODEL_VARIABLES,tf.GraphKeys.MOVING_AVERAGE_VARIABLES]

    #Define variables to keep track of the mean and variance
    moving_mean = tf.Variable(tf.zeros([num_channels]),name=(name+"/moving_mean"),trainable=False,collections=collections)
    moving_var = tf.Variable(tf.ones([num_channels]),name=(name+"/moving_variance"),trainable=False,collections=collections)
    beta = self.weight_variable_init_constant(name+"/beta", [tensor.shape[3].value], 0.0, reg=False)

    #This is the mean, computed only over exactly the areas of the mask, weighting each spot equally,
    #even across different elements in the batch that might have different board sizes.
    mean = tf.reduce_sum(tensor * mask,axis=[0,1,2]) / mask_sum
    zmtensor = tensor-mean
    #Similarly, the variance computed exactly only over those spots
    var = tf.reduce_sum(tf.square(zmtensor * mask),axis=[0,1,2]) / mask_sum
    mean_op = tf.keras.backend.moving_average_update(moving_mean,mean,0.998)
    var_op = tf.keras.backend.moving_average_update(moving_var,var,0.998)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_op)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, var_op)

    def training_f():
      return (mean,var)
    def inference_f():
      return (moving_mean,moving_var)

    use_mean,use_var = tf.cond(self.is_training_tensor,training_f,inference_f)
    return tf.nn.batch_normalization(tensor,use_mean,use_var,beta,None,epsilon) * mask

  # def batchnorm(self,name,tensor):
  #   epsilon = 0.001
  #   has_bias = True
  #   has_scale = False
  #   self.batch_norms[name] = (tensor.shape[-1].value,epsilon,has_bias,has_scale)
  #   return tf.layers.batch_normalization(
  #     tensor,
  #     axis=-1, #Because channels are our last axis, -1 refers to that via wacky python indexing
  #     momentum=0.99,
  #     epsilon=epsilon,
  #     center=has_bias,
  #     scale=has_scale,
  #     training=self.is_training_tensor,
  #     name=name,
  #   )

  def init_stdev(self,num_inputs,num_outputs):
    #xavier
    #return math.sqrt(2.0 / (num_inputs + num_outputs))
    #herangzhen
    return math.sqrt(2.0 / (num_inputs))

  def init_weights(self, shape, num_inputs, num_outputs):
    stdev = self.init_stdev(num_inputs,num_outputs) / 1.0
    return tf.truncated_normal(shape=shape, stddev=stdev)

  def weight_variable_init_constant(self, name, shape, constant, reg=True):
    init = tf.zeros(shape)
    if constant != 0.0:
      init = init + constant
    variable = tf.Variable(init,name=name)
    if reg is True:
      self.reg_variables.append(variable)
    elif reg == "tiny":
      self.reg_variables_tiny.append(variable)
    return variable

  def weight_variable(self, name, shape, num_inputs, num_outputs, scale_initial_weights=1.0, extra_initial_weight=None, reg=True):
    initial = self.init_weights(shape, num_inputs, num_outputs)
    if extra_initial_weight is not None:
      initial = initial + extra_initial_weight
    initial = initial * scale_initial_weights

    variable = tf.Variable(initial,name=name)
    if reg is True:
      self.reg_variables.append(variable)
    elif reg == "tiny":
      self.reg_variables_tiny.append(variable)
    return variable

  def conv2d(self, x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

  def dilated_conv2d(self, x, w, dilation):
    return tf.nn.atrous_conv2d(x, w, rate = dilation, padding='SAME')

  def apply_symmetry(self,tensor,symmetries,inverse):
    ud = symmetries[0]
    lr = symmetries[1]
    transp = symmetries[2]

    if not inverse:
      tensor = tf.cond(
        ud,
        lambda: tf.reverse(tensor,[1]),
        lambda: tensor
      )
      tensor = tf.cond(
        lr,
        lambda: tf.reverse(tensor,[2]),
        lambda: tensor
      )

    tensor = tf.cond(
      transp,
      lambda: tf.transpose(tensor, [0,2,1,3]),
      lambda: tensor)

    if inverse:
      tensor = tf.cond(
        ud,
        lambda: tf.reverse(tensor,[1]),
        lambda: tensor
      )
      tensor = tf.cond(
        lr,
        lambda: tf.reverse(tensor,[2]),
        lambda: tensor
      )

    return tensor

  #Define useful components --------------------------------------------------------------------------

  def relu(self, name, layer):
    assert(len(layer.shape) == 4)
    #num_channels = layer.shape[3].value
    #alphas = self.weight_variable_init_constant(name+"/relu",[1,1,1,num_channels],constant=0.0)
    return tf.nn.relu(layer)

  def relu_spatial1d(self, name, layer):
    assert(len(layer.shape) == 3)
    #num_channels = layer.shape[1].value
    #alphas = self.weight_variable_init_constant(name+"/relu",[1,num_channels],constant=0.0)
    return tf.nn.relu(layer)

  def relu_non_spatial(self, name, layer):
    assert(len(layer.shape) == 2)
    #num_channels = layer.shape[1].value
    #alphas = self.weight_variable_init_constant(name+"/relu",[1,num_channels],constant=0.0)
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
  def conv_block(
      self, name, in_layer, mask, mask_sum, diam, in_channels, out_channels,
      scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None
  ):
    weights = self.conv_weight_variable(name+"/w", diam, diam, in_channels, out_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr)
    convolved = self.conv2d(in_layer, weights)
    self.outputs_by_layer.append((name+"/prenorm",convolved))
    out_layer = self.relu(name+"/relu",self.batchnorm_and_mask(name+"/norm",convolved,mask,mask_sum))
    self.outputs_by_layer.append((name,out_layer))
    return out_layer

  #Convolution only, no batch norm or nonlinearity
  def conv_only_block(
      self, name, in_layer, diam, in_channels, out_channels,
      scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None, reg=True
  ):
    weights = self.conv_weight_variable(name+"/w", diam, diam, in_channels, out_channels, scale_initial_weights, emphasize_center_weight, emphasize_center_lr, reg=reg)
    out_layer = self.conv2d(in_layer, weights)
    self.outputs_by_layer.append((name,out_layer))
    return out_layer

  #Convolutional residual block with internal batch norm and nonlinear activation
  def res_conv_block(
      self, name, in_layer, mask, mask_sum, diam, main_channels, mid_channels,
      scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None
  ):
    trans1_layer = self.relu(name+"/relu1",(self.batchnorm_and_mask(name+"/norm1",in_layer,mask,mask_sum)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    fixup_scale = 1.0 / math.sqrt(self.num_blocks) if self.use_fixup else 1.0
    weights1 = self.conv_weight_variable(name+"/w1", diam, diam, main_channels, mid_channels, scale_initial_weights * fixup_scale, emphasize_center_weight, emphasize_center_lr)
    conv1_layer = self.conv2d(trans1_layer, weights1)
    self.outputs_by_layer.append((name+"/conv1",conv1_layer))

    trans2_layer = self.relu(name+"/relu2",(self.batchnorm_and_mask(name+"/norm2",conv1_layer,mask,mask_sum,use_gamma_in_fixup=True)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    fixup_scale_last_layer = 0.0 if self.use_fixup else 1.0
    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels, main_channels, scale_initial_weights*fixup_scale_last_layer, emphasize_center_weight, emphasize_center_lr)
    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer

  #Convolutional residual block with internal batch norm and nonlinear activation
  def global_res_conv_block(
      self, name, in_layer, mask, mask_sum, mask_sum_hw, mask_sum_hw_sqrt,
      diam, main_channels, mid_channels, global_mid_channels,
      scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None
  ):
    trans1_layer = self.relu(name+"/relu1",(self.batchnorm_and_mask(name+"/norm1",in_layer,mask,mask_sum)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    fixup_scale2 = 1.0 / math.sqrt(self.num_blocks) if self.use_fixup else 1.0
    fixup_scale4 = 1.0 / (self.num_blocks ** (1.0 / 4.0)) if self.use_fixup else 1.0
    weights1a = self.conv_weight_variable(name+"/w1a", diam, diam, main_channels, mid_channels, scale_initial_weights * fixup_scale2, emphasize_center_weight, emphasize_center_lr)
    weights1b = self.conv_weight_variable(name+"/w1b", diam, diam, main_channels, global_mid_channels, scale_initial_weights * fixup_scale4, emphasize_center_weight, emphasize_center_lr)
    conv1a_layer = self.conv2d(trans1_layer, weights1a)
    conv1b_layer = self.conv2d(trans1_layer, weights1b)
    self.outputs_by_layer.append((name+"/conv1a",conv1a_layer))
    self.outputs_by_layer.append((name+"/conv1b",conv1b_layer))

    trans1b_layer = self.relu(name+"/trans1b",(self.batchnorm_and_mask(name+"/norm1b",conv1b_layer,mask,mask_sum)))
    trans1b_pooled = self.global_pool(trans1b_layer, mask_sum_hw, mask_sum_hw_sqrt)

    remix_weights = self.weight_variable(name+"/w1r",[global_mid_channels*3,mid_channels],global_mid_channels*3,mid_channels, scale_initial_weights * fixup_scale4 * 0.5)
    conv1_layer = conv1a_layer + tf.tensordot(trans1b_pooled,remix_weights,axes=[[3],[0]])

    trans2_layer = self.relu(name+"/relu2",(self.batchnorm_and_mask(name+"/norm2",conv1_layer,mask,mask_sum,use_gamma_in_fixup=True)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    fixup_scale_last_layer = 0.0 if self.use_fixup else 1.0
    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels, main_channels, scale_initial_weights * fixup_scale_last_layer, emphasize_center_weight, emphasize_center_lr)
    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer

  #Convolutional residual block with internal batch norm and nonlinear activation
  def dilated_res_conv_block(self, name, in_layer, mask, mask_sum, diam, main_channels, mid_channels, dilated_mid_channels, dilation, scale_initial_weights=1.0, emphasize_center_weight=None, emphasize_center_lr=None):
    trans1_layer = self.relu(name+"/relu1",(self.batchnorm_and_mask(name+"/norm1",in_layer,mask,mask_sum)))
    self.outputs_by_layer.append((name+"/trans1",trans1_layer))

    fixup_scale = 1.0 / math.sqrt(self.num_blocks) if self.use_fixup else 1.0
    weights1a = self.conv_weight_variable(name+"/w1a", diam, diam, main_channels, mid_channels, scale_initial_weights*fixup_scale, emphasize_center_weight, emphasize_center_lr)
    weights1b = self.conv_weight_variable(name+"/w1b", diam, diam, main_channels, dilated_mid_channels, scale_initial_weights*fixup_scale, emphasize_center_weight, emphasize_center_lr)
    conv1a_layer = self.conv2d(trans1_layer, weights1a)
    conv1b_layer = self.dilated_conv2d(trans1_layer, weights1b, dilation=dilation)
    self.outputs_by_layer.append((name+"/conv1a",conv1a_layer))
    self.outputs_by_layer.append((name+"/conv1b",conv1b_layer))

    conv1_layer = tf.concat([conv1a_layer,conv1b_layer],axis=3)

    trans2_layer = self.relu(name+"/relu2",(self.batchnorm_and_mask(name+"/norm2",conv1_layer,mask,mask_sum,use_gamma_in_fixup=True)))
    self.outputs_by_layer.append((name+"/trans2",trans2_layer))

    fixup_scale_last_layer = 0.0 if self.use_fixup else 1.0
    weights2 = self.conv_weight_variable(name+"/w2", diam, diam, mid_channels+dilated_mid_channels, main_channels, scale_initial_weights * fixup_scale_last_layer, emphasize_center_weight, emphasize_center_lr)
    conv2_layer = self.conv2d(trans2_layer, weights2)
    self.outputs_by_layer.append((name+"/conv2",conv2_layer))

    return conv2_layer

  def global_pool(self, in_layer, mask_sum_hw, mask_sum_hw_sqrt):
    div = tf.reshape(mask_sum_hw,[-1,1,1,1])
    div_sqrt = tf.reshape(mask_sum_hw_sqrt,[-1,1,1,1])

    layer_raw_mean = tf.reduce_sum(in_layer,axis=[1,2],keepdims=True) / div
    layer_raw_max = tf.reduce_max(in_layer,axis=[1,2],keepdims=True)

    # 1, (x-14)/10, and (x-14)^2/100 - 0.1 are three orthogonal functions over [9,19], the range of reasonable board sizes.
    # We have the 14 in there since it's the midpoint of that range. The /10 is just sort of arbitrary normalization to keep things on the same scale.
    center_bsize = 14.0
    layer_0 = layer_raw_mean
    layer_1 = layer_raw_mean * ((div_sqrt - center_bsize) / 10.0)
    layer_2 = layer_raw_max

    layer_pooled = tf.concat([layer_0,layer_1,layer_2],axis=3)
    return layer_pooled

  def value_head_pool(self, in_layer, mask_sum_hw, mask_sum_hw_sqrt):
    div = tf.reshape(mask_sum_hw,[-1,1])
    div_sqrt = tf.reshape(mask_sum_hw_sqrt,[-1,1])

    layer_raw_mean = tf.reduce_sum(in_layer,axis=[1,2],keepdims=False) / div

    # 1, (x-14)/10, and (x-14)^2/100 - 0.1 are three orthogonal functions over [9,19], the range of reasonable board sizes.
    # We have the 14 in there since it's the midpoint of that range. The /10 and /100 are just sort of arbitrary normalization to keep things on the same scale
    center_bsize = 14.0
    layer_0 = layer_raw_mean
    layer_1 = layer_raw_mean * ((div_sqrt - center_bsize) / 10.0)
    layer_2 = layer_raw_mean * (tf.square(div_sqrt - center_bsize) / 100.0 - 0.1)
    layer_pooled = tf.concat([layer_0,layer_1,layer_2],axis=1)
    return layer_pooled


  #Begin Neural net------------------------------------------------------------------------------------
  #Indexing:
  #batch, bsize, bsize, channel

  def build_model(self,config,placeholders):
    pos_len = self.pos_len

    #Model version-------------------------------------------------------------------------------
    #This is written out in the model file when it gets built for export
    #self.version = 0 #V1 features, with old head architecture using crelus (no longer supported)
    #self.version = 1 #V1 features, with new head architecture, no crelus
    #self.version = 2 #V2 features, no internal architecture change.
    #self.version = 3 #V3 features, selfplay-planned features with lots of aux targets
    #self.version = 4 #V3 features, but supporting belief stdev and dynamic scorevalue
    #self.version = 5 #V4 features, slightly different pass-alive stones feature
    #self.version = 6 #V5 features, most higher-level go features removed
    #self.version = 7 #V6 features, more rules support
    #self.version = 8 #V7 features, asym

    self.version = Model.get_version(config)
    #These are the only four supported versions
    assert(self.version == 4 or self.version == 5 or self.version == 6 or self.version == 7 or self.version == 8)

    #Input layer---------------------------------------------------------------------------------
    bin_inputs = (placeholders["bin_inputs"] if "bin_inputs" in placeholders else
                  tf.placeholder(tf.float32, [None] + self.bin_input_shape, name="bin_inputs"))
    global_inputs = (placeholders["global_inputs"] if "global_inputs" in placeholders else
                    tf.placeholder(tf.float32, [None] + self.global_input_shape, name="global_inputs"))
    symmetries = (placeholders["symmetries"] if "symmetries" in placeholders else
                  tf.placeholder(tf.bool, [3], name="symmetries"))
    include_history = (placeholders["include_history"] if "include_history" in placeholders else
                       tf.placeholder(tf.float32, [None] + [5], name="include_history"))

    self.assert_batched_shape("bin_inputs",bin_inputs,self.bin_input_shape)
    self.assert_batched_shape("global_inputs",global_inputs,self.global_input_shape)
    self.assert_shape("symmetries",symmetries,[3])
    self.assert_batched_shape("include_history",include_history,[5])

    self.bin_inputs = bin_inputs
    self.global_inputs = global_inputs
    self.symmetries = symmetries
    self.include_history = include_history

    cur_layer = tf.reshape(bin_inputs, [-1] + self.post_input_shape)

    input_num_channels = self.post_input_shape[2]

    mask_before_symmetry = cur_layer[:,:,:,0:1]

    #Input symmetries - we apply symmetries during training by transforming the input and reverse-transforming the output
    cur_layer = self.apply_symmetry(cur_layer,symmetries,inverse=False)

    # #Disable various features
    # features_active = tf.constant([
    #   1.0, #0
    #   1.0, #1
    #   1.0, #2
    #   1.0, #3
    #   1.0, #4
    #   1.0, #5
    #   1.0, #6
    #   1.0, #7
    #   1.0, #8
    #   1.0, #9
    #   1.0, #10
    #   1.0, #11
    #   1.0, #12
    #   1.0, #13
    #   1.0, #14
    #   1.0, #15
    #   1.0, #16
    # ])
    # assert(features_active.dtype == tf.float32)
    # cur_layer = cur_layer * tf.reshape(features_active,[1,1,1,-1])

    #Apply history transform to turn off various features randomly.
    #We do this by building a matrix for each batch element, mapping input channels to possibly-turned off channels.
    #This matrix is a sum of hist_matrix_base which always turns off all the channels, and h0, h1, h2,... which perform
    #the modifications to hist_matrix_base to make it turn on channels based on whether we have move0, move1,...
    if self.version == 4 or self.version == 5 or self.version == 7 or self.version == 8:
      hist_matrix_base = np.diag(np.array([
        1.0, #0
        1.0, #1
        1.0, #2
        1.0, #3
        1.0, #4
        1.0, #5
        1.0, #6
        1.0, #7
        1.0, #8
        0.0, #9
        0.0, #10
        0.0, #11
        0.0, #12
        0.0, #13
        1.0, #14
        0.0, #15
        0.0, #16
        1.0, #17
        1.0, #18
        1.0, #19
        1.0, #20
        1.0, #21
      ],dtype=np.float32))
      #Because we have ladder features that express past states rather than past diffs, the most natural encoding when
      #we have no history is that they were always the same, rather than that they were all zero. So rather than zeroing
      #them we have no history, we add entries in the matrix to copy them over.
      #By default, without history, the ladder features 15 and 16 just copy over from 14.
      hist_matrix_base[14,15] = 1.0
      hist_matrix_base[14,16] = 1.0
      #When have the prev move, we enable feature 9 and 15
      h0 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h0[9,9] = 1.0 #Enable 9 -> 9
      h0[14,15] = -1.0 #Stop copying 14 -> 15
      h0[14,16] = -1.0 #Stop copying 14 -> 16
      h0[15,15] = 1.0 #Enable 15 -> 15
      h0[15,16] = 1.0 #Start copying 15 -> 16
      #When have the prevprev move, we enable feature 10 and 16
      h1 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h1[10,10] = 1.0 #Enable 10 -> 10
      h1[15,16] = -1.0 #Stop copying 15 -> 16
      h1[16,16] = 1.0 #Enable 16 -> 16
      #Further history moves
      h2 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h2[11,11] = 1.0
      h3 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h3[12,12] = 1.0
      h4 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h4[13,13] = 1.0
    elif self.version == 6:
      hist_matrix_base = np.diag(np.array([
        1.0, #0
        1.0, #1
        1.0, #2
        1.0, #3
        1.0, #4
        1.0, #5
        0.0, #6
        0.0, #7
        0.0, #8
        0.0, #9
        0.0, #10
        1.0, #11
        1.0, #12
      ],dtype=np.float32))
      h0 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h0[6,6] = 1.0
      h1 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h1[7,7] = 1.0
      h2 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h2[8,8] = 1.0
      h3 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h3[9,9] = 1.0
      h4 = np.zeros([self.num_bin_input_features,self.num_bin_input_features],dtype=np.float32)
      h4[10,10] = 1.0

    hist_matrix_base = tf.reshape(tf.constant(hist_matrix_base),[1,self.num_bin_input_features,self.num_bin_input_features])
    hist_matrix_builder = tf.constant(np.array([h0,h1,h2,h3,h4]))
    assert(hist_matrix_base.dtype == tf.float32)
    assert(hist_matrix_builder.dtype == tf.float32)
    assert(len(hist_matrix_builder.shape) == 3)
    assert(hist_matrix_builder.shape[0].value == 5)
    assert(hist_matrix_builder.shape[1].value == self.num_bin_input_features)
    assert(hist_matrix_builder.shape[2].value == self.num_bin_input_features)

    hist_filter_matrix = hist_matrix_base + tf.tensordot(include_history, hist_matrix_builder, axes=[[1],[0]]) #[batch,move] * [move,inc,outc] = [batch,inc,outc]
    cur_layer = tf.reshape(cur_layer,[-1,self.pos_len*self.pos_len,self.num_bin_input_features]) #[batch,xy,inc]
    cur_layer = tf.matmul(cur_layer,hist_filter_matrix) #[batch,xy,inc] * [batch,inc,outc] = [batch,xy,outc]
    cur_layer = tf.reshape(cur_layer,[-1,self.pos_len,self.pos_len,self.num_bin_input_features])

    assert(include_history.shape[1].value == 5)
    transformed_global_inputs = global_inputs * tf.pad(include_history, [(0,0),(0,self.num_global_input_features - include_history.shape[1].value)], constant_values=1.0)

    self.transformed_bin_inputs = cur_layer
    self.transformed_global_inputs = transformed_global_inputs

    #Channel counts---------------------------------------------------------------------------------------
    trunk_num_channels = config["trunk_num_channels"]
    mid_num_channels = config["mid_num_channels"]
    regular_num_channels = config["regular_num_channels"]
    dilated_num_channels = config["dilated_num_channels"]
    gpool_num_channels = config["gpool_num_channels"]

    assert(regular_num_channels + dilated_num_channels == mid_num_channels)

    self.trunk_num_channels = trunk_num_channels
    self.mid_num_channels = mid_num_channels
    self.regular_num_channels = regular_num_channels
    self.dilated_num_channels = dilated_num_channels
    self.gpool_num_channels = gpool_num_channels

    mask = cur_layer[:,:,:,0:1]
    mask_sum = tf.reduce_sum(mask) # Global sum
    mask_sum_hw = tf.reduce_sum(mask,axis=[1,2,3]) # Sum per batch element
    mask_sum_hw_sqrt = tf.sqrt(mask_sum_hw)

    #Initial convolutional layer-------------------------------------------------------------------------------------
    trunk = self.conv_only_block("conv1",cur_layer,diam=5,in_channels=input_num_channels,out_channels=trunk_num_channels)
    self.initial_conv = ("conv1",5,input_num_channels,trunk_num_channels)

    #Matrix multiply global inputs and accumulate them
    ginputw = self.weight_variable("ginputw",[self.num_global_input_features,trunk_num_channels],self.num_global_input_features*2,trunk_num_channels)
    ginputresult = tf.tensordot(transformed_global_inputs,ginputw,axes=[[1],[0]])
    trunk = trunk + tf.reshape(ginputresult, [-1,1,1,trunk_num_channels])

    self.initial_matmul = ("ginputw",self.num_global_input_features,trunk_num_channels)

    #Main trunk---------------------------------------------------------------------------------------------------
    self.blocks = []

    block_kind = config["block_kind"]

    for i in range(len(block_kind)):
      (name,kind) = block_kind[i]
      if kind == "regular":
        residual = self.res_conv_block(
          name,trunk,mask,mask_sum,diam=3,main_channels=trunk_num_channels,mid_channels=mid_num_channels)
        trunk = self.merge_residual(name,trunk,residual)
        self.blocks.append(("ordinary_block",name,3,trunk_num_channels,mid_num_channels))
      elif kind == "dilated":
        residual = self.dilated_res_conv_block(
          name,trunk,mask,mask_sum,diam=3,main_channels=trunk_num_channels,mid_channels=regular_num_channels, dilated_mid_channels=dilated_num_channels, dilation=2
        )
        trunk = self.merge_residual(name,trunk,residual)
        self.blocks.append(("dilated_block",name,3,trunk_num_channels,regular_num_channels,dilated_num_channels,3))
      elif kind == "gpool":
        residual = self.global_res_conv_block(
          name,trunk,mask,mask_sum,mask_sum_hw,mask_sum_hw_sqrt,
          diam=3,main_channels=trunk_num_channels,mid_channels=regular_num_channels, global_mid_channels=gpool_num_channels
        )
        trunk = self.merge_residual(name,trunk,residual)
        self.blocks.append(("gpool_block",name,3,trunk_num_channels,regular_num_channels,gpool_num_channels))
      else:
        assert(False)

    #Postprocessing residual trunk----------------------------------------------------------------------------------

    #Normalize and relu just before the policy head
    trunk = self.relu("trunk/relu",(self.batchnorm_and_mask("trunk/norm",trunk,mask,mask_sum)))
    self.outputs_by_layer.append(("trunk",trunk))

    #Policy head---------------------------------------------------------------------------------
    p0_layer = trunk

    #This is the main path for policy information
    p1_num_channels = config["p1_num_channels"]
    p1_intermediate_conv = self.conv_only_block("p1/intermediate_conv",p0_layer,diam=1,in_channels=trunk_num_channels,out_channels=p1_num_channels)
    self.p1_conv = ("p1/intermediate_conv",1,trunk_num_channels,p1_num_channels)

    #But in parallel convolve to compute some features about the global state of the board
    #Hopefully the neural net uses this for stuff like ko situation, overall temperature/threatyness, who is leading, etc.
    g1_num_channels = config["g1_num_channels"]
    g1_layer = self.conv_block("g1",p0_layer,mask,mask_sum,diam=1,in_channels=trunk_num_channels,out_channels=g1_num_channels)
    self.g1_conv = ("g1",1,trunk_num_channels,g1_num_channels)

    #Fold g1 down to single values for the board.
    g2_layer = self.global_pool(g1_layer, mask_sum_hw, mask_sum_hw_sqrt) #shape [b,1,1,3*g1_num_channels]
    g2_num_channels = 3*g1_num_channels
    self.outputs_by_layer.append(("g2",g2_layer))

    #Transform them into the space of the policy features to act as biases for the policy
    #Also divide the initial weights a bit more because we think these should matter a bit less than local shape stuff,
    #by multiplying the number of inputs for purposes of weight initialization (currently mult by 4)
    matmulg2w = self.weight_variable("matmulg2w",[g2_num_channels,p1_num_channels],g2_num_channels*4,p1_num_channels)
    g3_layer = tf.tensordot(g2_layer,matmulg2w,axes=[[3],[0]])
    self.outputs_by_layer.append(("g3",g3_layer))
    self.g1_num_channels = g1_num_channels
    self.g2_num_channels = g2_num_channels
    self.p1_num_channels = p1_num_channels

    #Add! This adds shapes [b,19,19,convp1_num_channels] + [b,1,1,convp1_num_channels]
    #so the second one should get broadcast up to the size of the first one.
    #We can think of p1 as being an ordinary convolution layer except that for every node of the convolution, the g2 values (g2_num_channels many of them)
    #have been appended to the p0 incoming values (p0_num_channels * convp1diam * convp1diam many of them).
    #The matrix matmulg2w is simply the set of weights for that additional part of the matrix. It's just that rather than appending beforehand,
    #we multiply separately and add to the output afterward.
    p1_intermediate_sum = p1_intermediate_conv + g3_layer

    #And now apply batchnorm and relu
    p1_layer = self.relu("p1/relu",self.batchnorm_and_mask("p1/norm",p1_intermediate_sum,mask,mask_sum))
    self.outputs_by_layer.append(("p1",p1_layer))

    #Finally, apply linear convolution to produce final output
    p2_layer = self.conv_only_block("p2",p1_layer,diam=1,in_channels=p1_num_channels,out_channels=2,scale_initial_weights=0.3)
    p2_layer = p2_layer - (1.0-mask) * 5000.0 # mask out parts outside the board by making them a huge neg number, so that they're 0 after softmax
    self.p2_conv = ("p2",1,p1_num_channels,2)

    # self.add_lr_factor("p1/norm/beta:0",0.25)
    # self.add_lr_factor("p2/w:0",0.25)

    #Output symmetries - we apply symmetries during training by transforming the input and reverse-transforming the output
    policy_output = self.apply_symmetry(p2_layer,symmetries,inverse=True)
    policy_output = tf.reshape(policy_output, [-1] + self.policy_output_shape_nopass)

    #Add pass move based on the global g values
    matmulpass = self.weight_variable("matmulpass",[g2_num_channels,2],g2_num_channels*8,2,scale_initial_weights=0.3)
    # self.add_lr_factor("matmulpass:0",0.25)
    pass_output = tf.tensordot(g2_layer,matmulpass,axes=[[3],[0]])
    self.outputs_by_layer.append(("pass",pass_output))
    pass_output = tf.reshape(pass_output, [-1] + [1,2])
    policy_output = tf.concat([policy_output,pass_output],axis=1, name="policy_output")

    self.policy_output = policy_output

    #Value head---------------------------------------------------------------------------------
    v0_layer = trunk

    v1_num_channels = config["v1_num_channels"]
    v1_layer = self.conv_block("v1",v0_layer,mask,mask_sum,diam=1,in_channels=trunk_num_channels,out_channels=v1_num_channels)
    self.outputs_by_layer.append(("v1",v1_layer))
    self.v1_conv = ("v1",1,trunk_num_channels,v1_num_channels)
    self.v1_num_channels = v1_num_channels

    v1_layer_pooled = self.value_head_pool(v1_layer, mask_sum_hw, mask_sum_hw_sqrt)
    v1_size = v1_num_channels

    v2_size = config["v2_size"]
    v2w = self.weight_variable("v2/w",[v1_size*3,v2_size],v1_size*3,v2_size)
    v2b = self.weight_variable("v2/b",[v2_size],v1_size*3,v2_size,scale_initial_weights=0.2,reg="tiny")
    v2_layer = self.relu_non_spatial("v2/relu",tf.matmul(v1_layer_pooled, v2w) + v2b)
    self.v2_size = v2_size
    self.other_internal_outputs.append(("v2",v2_layer))

    v3_size = self.value_target_shape[0]
    v3w = self.weight_variable("v3/w",[v2_size,v3_size],v2_size,v3_size)
    v3b = self.weight_variable("v3/b",[v3_size],v2_size,v3_size,scale_initial_weights=0.2,reg="tiny")
    v3_layer = tf.matmul(v2_layer, v3w) + v3b
    self.v3_size = v3_size
    self.other_internal_outputs.append(("v3",v3_layer))

    mv3_size = self.miscvalues_target_shape[0]
    mv3w = self.weight_variable("mv3/w",[v2_size,mv3_size],v2_size,mv3_size)
    mv3b = self.weight_variable("mv3/b",[mv3_size],v2_size,mv3_size,scale_initial_weights=0.2,reg="tiny")
    mv3_layer = tf.matmul(v2_layer, mv3w) + mv3b
    self.mv3_size = mv3_size
    self.other_internal_outputs.append(("mv3",mv3_layer))

    if not self.support_japanese_rules:
      # Force no-result prediction to 0 after softmax
      v3_layer = v3_layer + tf.constant([0,0,-5000.0],dtype=tf.float32)
    value_output = tf.reshape(v3_layer, [-1] + self.value_target_shape, name = "value_output")

    miscvalues_output = tf.reshape(mv3_layer, [-1] + self.miscvalues_target_shape, name = "miscvalues_output")

    #Transform a real-valued output into a positive value suitable for multiplying to other inputs as a scaling factor
    def scaletransform(tensor):
      self.prescale_variables.append(tensor)

      #tf.where has a bug where nan values on the non-chosen side will still propagate nans back in gradients.
      #So we also abs the tensor, so that we never get a log of a negative value
      abstensor = tf.abs(tensor)
      return tf.where(tensor > 0, 1.0 + tf.log(abstensor + 1.0), 1.0 / (1.0 + tf.log(abstensor + 1.0)))

    scorebelief_len = self.scorebelief_target_shape[0]
    scorebelief_mid = self.pos_len*self.pos_len+Model.EXTRA_SCORE_DISTR_RADIUS
    assert(scorebelief_len == self.pos_len*self.pos_len*2+Model.EXTRA_SCORE_DISTR_RADIUS*2)

    if self.version == 4 or self.version == 5 or self.version == 7 or self.version == 8:
      self.score_belief_offset_vector = np.array([float(i-scorebelief_mid)+0.5 for i in range(scorebelief_len)],dtype=np.float32)
      self.score_belief_parity_vector = np.array([0.5-float((i-scorebelief_mid) % 2) for i in range(scorebelief_len)],dtype=np.float32)
      sbv2_size = config["sbv2_num_channels"]
      sb2w = self.weight_variable("sb2/w",[v1_size*3,sbv2_size],v1_size*3+1,sbv2_size)
      sb2b = self.weight_variable("sb2/b",[sbv2_size],v1_size*3+1,sbv2_size,scale_initial_weights=0.2,reg="tiny")
      sb2_layer_partial = tf.matmul(v1_layer_pooled, sb2w) + sb2b
      sb2_offset_vector = tf.constant(0.05 * self.score_belief_offset_vector, dtype=tf.float32)
      sb2_parity_vector = tf.reshape(self.score_belief_parity_vector,[1,-1]) * transformed_global_inputs[:,self.num_global_input_features-1:self.num_global_input_features]
      sb2_offset_w = self.weight_variable("sb2_offset/w",[1,sbv2_size],v1_size*3+1,sbv2_size,scale_initial_weights=0.5)
      sb2_offset_partial = tf.matmul(tf.reshape(sb2_offset_vector,[-1,1]), sb2_offset_w)
      sb2_parity_w = self.weight_variable("sb2_parity/w",[1,sbv2_size],v1_size*3+1,sbv2_size)
      sb2_parity_partial = tf.matmul(tf.reshape(sb2_parity_vector,[-1,1]), sb2_parity_w)
      sb2_layer = (
        tf.reshape(sb2_layer_partial,[-1,1,sbv2_size]) +
        tf.reshape(sb2_offset_partial,[1,scorebelief_len,sbv2_size]) +
        tf.reshape(sb2_parity_partial,[-1,scorebelief_len,sbv2_size])
      )
      sb2_layer = self.relu_spatial1d("sb2/relu",sb2_layer)
    elif self.version == 6:
      self.score_belief_offset_vector = np.array([float(i-scorebelief_mid)+0.5 for i in range(scorebelief_len)],dtype=np.float32)
      sbv2_size = config["sbv2_num_channels"]
      sb2w = self.weight_variable("sb2/w",[v1_size*3,sbv2_size],v1_size*3+1,sbv2_size)
      sb2b = self.weight_variable("sb2/b",[sbv2_size],v1_size*3+1,sbv2_size,scale_initial_weights=0.2,reg="tiny")
      sb2_layer_partial = tf.matmul(v1_layer_pooled, sb2w) + sb2b
      sb2_offset_vector = tf.constant(0.05 * self.score_belief_offset_vector, dtype=tf.float32)
      sb2_offset_w = self.weight_variable("sb2_offset/w",[1,sbv2_size],v1_size*3+1,sbv2_size,scale_initial_weights=0.5)
      sb2_offset_partial = tf.matmul(tf.reshape(sb2_offset_vector,[-1,1]), sb2_offset_w)
      sb2_layer = (
        tf.reshape(sb2_layer_partial,[-1,1,sbv2_size]) +
        tf.reshape(sb2_offset_partial,[1,scorebelief_len,sbv2_size])
      )
      sb2_layer = self.relu_spatial1d("sb2/relu",sb2_layer)
    else:
      assert(False)

    sbscale2w = self.weight_variable("sbscale2/w",[v1_size*3,sbv2_size],v1_size*3+1,sbv2_size,scale_initial_weights=0.5)
    sbscale2b = self.weight_variable("sbscale2/b",[sbv2_size],v1_size*3+1,sbv2_size,scale_initial_weights=0.2,reg="tiny")
    sbscale2_layer = self.relu_non_spatial("sbscale2/relu",tf.matmul(v1_layer_pooled, sbscale2w) + sbscale2b)

    sb3w = self.weight_variable("sb3/w",[sbv2_size,1],sbv2_size,1,scale_initial_weights=0.5)
    sb3_layer = tf.tensordot(sb2_layer,sb3w,axes=[[2],[0]])

    sbscale3w = self.weight_variable("sbscale3/w",[sbv2_size,1],sbv2_size,1,scale_initial_weights=0.5)
    sbscale3_layer = scaletransform(tf.matmul(sbscale2_layer,sb3w))
    self.sbscale3_layer = sbscale3_layer

    sb3_layer = sb3_layer * tf.reshape(sbscale3_layer,[-1,1,1])

    scorebelief_output = tf.reshape(sb3_layer,[-1] + self.scorebelief_target_shape, name = "scorebelief_output")

    #No need for separate mask since v1_layer is already zero outside of mask bounds.
    ownership_output = self.conv_only_block("vownership",v1_layer,diam=1,in_channels=v1_num_channels,out_channels=1, scale_initial_weights=0.2) * mask
    self.vownership_conv = ("vownership",1,v1_num_channels,1)
    ownership_output = self.apply_symmetry(ownership_output,symmetries,inverse=True)
    ownership_output = tf.reshape(ownership_output, [-1] + self.ownership_target_shape, name = "ownership_output")

    scoring_output = self.conv_only_block("vscoring",v1_layer,diam=1,in_channels=v1_num_channels,out_channels=1, scale_initial_weights=0.2) * mask
    self.vscoring_conv = ("vscoring",1,v1_num_channels,1)
    scoring_output = self.apply_symmetry(scoring_output,symmetries,inverse=True)
    scoring_output = tf.reshape(scoring_output, [-1] + self.scoring_target_shape, name = "scoring_output")

    futurepos_output = self.conv_only_block("futurepos",v0_layer,diam=1,in_channels=trunk_num_channels,out_channels=2, scale_initial_weights=0.2) * mask
    self.futurepos_conv = ("futurepos",1,trunk_num_channels,2)
    futurepos_output = self.apply_symmetry(futurepos_output,symmetries,inverse=True)
    futurepos_output = tf.reshape(futurepos_output, [-1] + self.futurepos_target_shape, name = "futurepos_output")

    seki_output = self.conv_only_block("seki",v0_layer,diam=1,in_channels=trunk_num_channels,out_channels=4, scale_initial_weights=0.2) * mask
    self.seki_conv = ("seki",1,trunk_num_channels,4)
    seki_output = self.apply_symmetry(seki_output,symmetries,inverse=True)
    seki_output = tf.reshape(seki_output, [-1] + self.seki_output_shape, name = "seki_output")

    # self.add_lr_factor("v2/w:0",0.25)
    # self.add_lr_factor("v2/b:0",0.25)
    # self.add_lr_factor("v3/w:0",0.25)
    # self.add_lr_factor("v3/b:0",0.25)
    # self.add_lr_factor("mv3/w:0",0.25)
    # self.add_lr_factor("mv3/b:0",0.25)
    # self.add_lr_factor("sb2/w:0",0.25)
    # self.add_lr_factor("sb2/b:0",0.25)
    # self.add_lr_factor("sb2_offset/w:0",0.25)
    # # self.add_lr_factor("sbscale2/w:0",0.25)
    # # self.add_lr_factor("sbscale2/b:0",0.25)
    # self.add_lr_factor("sb3/w:0",0.25)
    # # self.add_lr_factor("sbscale3/w:0",0.25)
    # self.add_lr_factor("vownership/w:0",0.25)
    # self.add_lr_factor("vscoring/w:0",0.25)

    self.value_output = value_output
    self.miscvalues_output = miscvalues_output
    self.scorebelief_output = scorebelief_output
    self.ownership_output = ownership_output
    self.scoring_output = scoring_output
    self.futurepos_output = futurepos_output
    self.seki_output = seki_output

    self.mask_before_symmetry = mask_before_symmetry
    self.mask = mask
    self.mask_sum = mask_sum
    self.mask_sum_hw = mask_sum_hw
    self.mask_sum_hw_sqrt = mask_sum_hw_sqrt

class Target_vars:
  def __init__(self,model,for_optimization,placeholders):
    policy_output = model.policy_output
    value_output = model.value_output
    miscvalues_output = model.miscvalues_output
    scorebelief_output = model.scorebelief_output
    ownership_output = model.ownership_output
    scoring_output = model.scoring_output
    futurepos_output = model.futurepos_output
    seki_output = model.seki_output

    value_probs = tf.nn.softmax(value_output,axis=1)
    scorebelief_probs = tf.nn.softmax(scorebelief_output,axis=1)

    scoremean_prediction = miscvalues_output[:,0] * 20.0
    scorestdev_prediction = tf.math.softplus(miscvalues_output[:,1]) * 20.0
    lead_prediction = miscvalues_output[:,2] * 20.0
    variance_time_prediction = tf.math.softplus(miscvalues_output[:,3]) * 150.0

    #Loss function
    self.policy_target = (placeholders["policy_target"] if "policy_target" in placeholders else
                          tf.placeholder(tf.float32, [None] + model.policy_target_shape))
    self.policy_target1 = (placeholders["policy_target1"] if "policy_target1" in placeholders else
                          tf.placeholder(tf.float32, [None] + model.policy_target_shape))
    #Unconditional game result prediction
    self.value_target = (placeholders["value_target"] if "value_target" in placeholders else
                         tf.placeholder(tf.float32, [None] + model.value_target_shape))
    self.td_value_target = (placeholders["td_value_target"] if "td_value_target" in placeholders else
                            tf.placeholder(tf.float32, [None] + model.td_value_target_shape))
    #Expected score prediction CONDITIONAL on result
    self.scoremean_target = (placeholders["scoremean_target"] if "scoremean_target" in placeholders else
                              tf.placeholder(tf.float32, [None] + model.scoremean_target_shape))
    self.lead_target = (placeholders["lead_target"] if "lead_target" in placeholders else
                              tf.placeholder(tf.float32, [None] + model.lead_target_shape))
    #Arrival time of variance in game, unconditional
    self.variance_time_target = (placeholders["variance_time_target"] if "variance_time_target" in placeholders else
                              tf.placeholder(tf.float32, [None] + model.variance_time_target_shape))
    #Score belief distributions CONDITIONAL on result
    self.scorebelief_target = (placeholders["scorebelief_target"] if "scorebelief_target" in placeholders else
                              tf.placeholder(tf.float32, [None] + model.scorebelief_target_shape))
    #Ownership of board, CONDITIONAL on result
    self.ownership_target = (placeholders["ownership_target"] if "ownership_target" in placeholders else
                             tf.placeholder(tf.float32, [None] + model.ownership_target_shape))
    #Scoring of board, CONDITIONAL on result
    self.scoring_target = (placeholders["scoring_target"] if "scoring_target" in placeholders else
                             tf.placeholder(tf.float32, [None] + model.scoring_target_shape))
    #Future board positions, unconditional
    self.futurepos_target = (placeholders["futurepos_target"] if "futurepos_target" in placeholders else
                             tf.placeholder(tf.float32, [None] + model.futurepos_target_shape))
    #Seki state of final board, CONDITIONAL on result
    self.seki_target = (placeholders["seki_target"] if "seki_target" in placeholders else
                             tf.placeholder(tf.float32, [None] + model.seki_target_shape))

    self.target_weight_from_data = (placeholders["target_weight_from_data"] if "target_weight_from_data" in placeholders else
                                    tf.placeholder(tf.float32, [None] + model.target_weight_shape))
    self.policy_target_weight = (placeholders["policy_target_weight"] if "policy_target_weight" in placeholders else
                                 tf.placeholder(tf.float32, [None] + model.policy_target_weight_shape))
    self.policy_target_weight1 = (placeholders["policy_target_weight1"] if "policy_target_weight1" in placeholders else
                                 tf.placeholder(tf.float32, [None] + model.policy_target_weight_shape))
    self.lead_target_weight = (placeholders["lead_target_weight"] if "lead_target_weight" in placeholders else
                                    tf.placeholder(tf.float32, [None] + model.lead_target_weight_shape))
    self.ownership_target_weight = (placeholders["ownership_target_weight"] if "ownership_target_weight" in placeholders else
                                    tf.placeholder(tf.float32, [None] + model.ownership_target_weight_shape))
    self.scoring_target_weight = (placeholders["scoring_target_weight"] if "scoring_target_weight" in placeholders else
                                    tf.placeholder(tf.float32, [None] + model.scoring_target_weight_shape))
    self.futurepos_target_weight = (placeholders["futurepos_target_weight"] if "futurepos_target_weight" in placeholders else
                                    tf.placeholder(tf.float32, [None] + model.futurepos_target_weight_shape))
    self.selfkomi = (placeholders["selfkomi"] if "selfkomi" in placeholders else
                     tf.placeholder(tf.float32, [None]))

    model.assert_batched_shape("policy_target", self.policy_target, model.policy_target_shape)
    model.assert_batched_shape("policy_target_weight", self.policy_target_weight, model.policy_target_weight_shape)
    model.assert_batched_shape("policy_target1", self.policy_target1, model.policy_target_shape)
    model.assert_batched_shape("policy_target_weight1", self.policy_target_weight1, model.policy_target_weight_shape)
    model.assert_batched_shape("value_target", self.value_target, model.value_target_shape)
    model.assert_batched_shape("td_value_target", self.td_value_target, model.td_value_target_shape)
    model.assert_batched_shape("scoremean_target", self.scoremean_target, model.scoremean_target_shape)
    model.assert_batched_shape("lead_target", self.lead_target, model.lead_target_shape)
    model.assert_batched_shape("variance_time_target", self.variance_time_target, model.variance_time_target_shape)
    model.assert_batched_shape("scorebelief_target", self.scorebelief_target, model.scorebelief_target_shape)
    model.assert_batched_shape("ownership_target", self.ownership_target, model.ownership_target_shape)
    model.assert_batched_shape("scoring_target", self.scoring_target, model.scoring_target_shape)
    model.assert_batched_shape("futurepos_target", self.futurepos_target, model.futurepos_target_shape)
    model.assert_batched_shape("seki_target", self.seki_target, model.seki_target_shape)
    model.assert_batched_shape("target_weight_from_data", self.target_weight_from_data, model.target_weight_shape)
    model.assert_batched_shape("lead_target_weight", self.lead_target_weight, model.lead_target_weight_shape)
    model.assert_batched_shape("ownership_target_weight", self.ownership_target_weight, model.ownership_target_weight_shape)
    model.assert_batched_shape("scoring_target_weight", self.scoring_target_weight, model.scoring_target_weight_shape)
    model.assert_batched_shape("futurepos_target_weight", self.futurepos_target_weight, model.futurepos_target_weight_shape)
    model.assert_batched_shape("selfkomi", self.selfkomi, [])

    self.target_weight_used = self.target_weight_from_data


    self.policy_loss_unreduced = self.policy_target_weight * (
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy_target, logits=policy_output[:,:,0])
    )
    self.policy1_loss_unreduced = self.policy_target_weight1 * 0.15 * (
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy_target1, logits=policy_output[:,:,1])
    )

    self.value_loss_unreduced = 1.20 * tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=self.value_target,
      logits=value_output
    )

    self.td_value_loss_unreduced = 0.60 * (
      tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=self.td_value_target,
        logits=tf.reshape(miscvalues_output[:,4:10],[-1] + model.td_value_target_shape)
      ) -
      # Subtract out the entropy, so as to get loss 0 at perfect prediction
      tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=self.td_value_target,
        logits=tf.log(self.td_value_target + 1.0e-30)
      )
    )
    self.td_value_loss_unreduced = tf.reduce_sum(self.td_value_loss_unreduced, axis=1)

    self.scorebelief_cdf_loss_unreduced = 0.015 * self.ownership_target_weight * (
      tf.reduce_sum(
        tf.square(tf.cumsum(self.scorebelief_target,axis=1) - tf.cumsum(tf.nn.softmax(scorebelief_output,axis=1),axis=1)),
        axis=1
      )
    )
    self.scorebelief_pdf_loss_unreduced = 0.015 * self.ownership_target_weight * (
      tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=self.scorebelief_target,
        logits=scorebelief_output
      )
    )

    #This uses a formulation where each batch element cares about its average loss.
    #In particular this means that ownership loss predictions on small boards "count more" per spot.
    #Not unlike the way that policy and value loss are also equal-weighted by batch element.
    self.ownership_loss_unreduced = 1.0 * self.ownership_target_weight * (
      tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=tf.stack([(1+self.ownership_target)/2,(1-self.ownership_target)/2],axis=3),
          logits=tf.stack([ownership_output,-ownership_output],axis=3)
        ) * tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len]),
        axis=[1,2]
      ) / model.mask_sum_hw
    )

    self.scoring_loss_unreduced = 0.6 * self.scoring_target_weight * (
      tf.reduce_sum(
        tf.square(self.scoring_target - scoring_output) * tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len]),
        axis=[1,2]
      ) / model.mask_sum_hw
    )

    #The futurepos targets extrapolate a fixed number of steps into the future independent
    #of board size. So unlike the ownership above, generally a fixed number of spots are going to be
    #"wrong" independent of board size, so we should just equal-weight the prediction per spot.
    #However, on larger boards often the entropy of where the future moves will be should be greater
    #and also in the event of capture, there may be large captures that don't occur on small boards,
    #causing some scaling with board size. So, I dunno, let's compromise and scale by sqrt(boardarea).
    #Also, the further out targets should be weighted a little less due to them being higher entropy
    #due to simply being farther in the future, so multiply by [1,0.25].
    self.futurepos_loss_unreduced = 0.20 * self.futurepos_target_weight * (
      tf.reduce_sum(
        tf.square(tf.tanh(futurepos_output) - self.futurepos_target)
        * tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len,1])
        * tf.reshape(tf.constant([1,0.25],dtype=tf.float32),[1,1,1,2]),
        axis=[1,2,3]
      ) / tf.sqrt(model.mask_sum_hw)
    )

    #Seki-lost-points target, same as ownership except lower weight
    owned_target = tf.square(self.ownership_target)
    unowned_target = 1.0 - owned_target
    unowned_proportion = (
      tf.reduce_sum(unowned_target * tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len]),axis=[1,2])
      / (1.0 + tf.reduce_sum(tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len]),axis=[1,2]))
    )
    unowned_proportion = tf.reduce_mean(unowned_proportion * self.ownership_target_weight)
    if model.is_training:
      moving_unowned_proportion = tf.Variable(1.0,name=("moving_unowned_proportion"),trainable=False)
      moving_unowned_op = tf.keras.backend.moving_average_update(moving_unowned_proportion,unowned_proportion,0.998)
      with tf.control_dependencies([moving_unowned_op]):
        seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
    else:
      seki_weight_scale = 1.0

    self.seki_loss_unreduced = (
      tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=tf.stack([1.0-tf.square(self.seki_target), tf.nn.relu(self.seki_target), tf.nn.relu(-self.seki_target)],axis=3),
          logits=seki_output[:,:,:,0:3]
        ) * tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len]),
        axis=[1,2]
      ) / model.mask_sum_hw
    )
    self.seki_loss_unreduced = self.seki_loss_unreduced + 0.5 * (
      tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=tf.stack([unowned_target, owned_target],axis=3),
          logits=tf.stack([seki_output[:,:,:,3],tf.zeros_like(self.ownership_target)],axis=3)
        ) * tf.reshape(model.mask_before_symmetry,[-1,model.pos_len,model.pos_len]),
        axis=[1,2]
      ) / model.mask_sum_hw
    )
    self.seki_loss_unreduced = seki_weight_scale * self.ownership_target_weight * self.seki_loss_unreduced
    self.seki_weight_scale = seki_weight_scale

    def huber_loss(x,y,delta):
      absdiff = tf.abs(x - y)
      return tf.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)

    #This is conditional upon there being a result
    expected_score_from_belief = tf.reduce_sum(scorebelief_probs * model.score_belief_offset_vector,axis=1)

    #Huber will incentivize this to not actually converge to the mean, but rather something meanlike locally and something medianlike
    #for very large possible losses. This seems... okay - it might actually be what users want.
    self.scoremean_loss_unreduced = 0.0012 * self.ownership_target_weight * huber_loss(self.scoremean_target, scoremean_prediction, delta = 12.0)
    self.lead_loss_unreduced = 0.022 * self.lead_target_weight * huber_loss(self.lead_target, lead_prediction, delta = 8.0)
    self.variance_time_loss_unreduced = 0.00000 * huber_loss(self.variance_time_target, variance_time_prediction, delta = 100.0)

    stdev_of_belief = tf.sqrt(0.001 + tf.reduce_sum(
      scorebelief_probs * tf.square(
        tf.reshape(model.score_belief_offset_vector,[1,-1]) - tf.reshape(expected_score_from_belief,[-1,1])
      ),axis=1))
    beliefstdevdiff = stdev_of_belief - scorestdev_prediction
    self.scorestdev_reg_loss_unreduced = 0.004 * huber_loss(stdev_of_belief, scorestdev_prediction, delta = 10.0)

    # winlossprob_from_belief = tf.concat([
    #   tf.reduce_sum(scorebelief_probs[:,(model.scorebelief_target_shape[0]//2):],axis=1,keepdims=True),
    #   tf.reduce_sum(scorebelief_probs[:,0:(model.scorebelief_target_shape[0]//2)],axis=1,keepdims=True)
    # ],axis=1) * (1.0 - tf.reshape(value_probs[:,2],[-1,1])) #Need to multiply here to convert conditional WL belief into unconditional, since noResult = 0
    # winlossprob_from_output = value_probs[:,0:2]
    # self.winloss_reg_loss_unreduced = 2.0 * tf.reduce_sum(tf.square(winlossprob_from_belief - winlossprob_from_output),axis=1)

    self.scale_reg_loss_unreduced = tf.reshape(0.0005 * tf.add_n([tf.square(variable) for variable in model.prescale_variables]), [-1])
    #self.scale_reg_loss_unreduced = tf.zeros_like(self.winloss_reg_loss_unreduced)

    self.policy_loss = tf.reduce_sum(self.target_weight_used * self.policy_loss_unreduced, name="losses/policy_loss")
    self.policy1_loss = tf.reduce_sum(self.target_weight_used * self.policy1_loss_unreduced, name="losses/policy1_loss")
    self.value_loss = tf.reduce_sum(self.target_weight_used * self.value_loss_unreduced, name="losses/value_loss")
    self.td_value_loss = tf.reduce_sum(self.target_weight_used * self.td_value_loss_unreduced, name="losses/td_value_loss")
    self.scoremean_loss = tf.reduce_sum(self.target_weight_used * self.scoremean_loss_unreduced, name="losses/scoremean_loss")
    self.lead_loss = tf.reduce_sum(self.target_weight_used * self.lead_loss_unreduced, name="losses/lead_loss")
    self.variance_time_loss = tf.reduce_sum(self.target_weight_used * self.variance_time_loss_unreduced, name="losses/variance_time_loss")
    self.scorebelief_pdf_loss = tf.reduce_sum(self.target_weight_used * self.scorebelief_pdf_loss_unreduced, name="losses/scorebelief_pdf_loss")
    self.scorebelief_cdf_loss = tf.reduce_sum(self.target_weight_used * self.scorebelief_cdf_loss_unreduced, name="losses/scorebelief_cdf_loss")
    self.ownership_loss = tf.reduce_sum(self.target_weight_used * self.ownership_loss_unreduced, name="losses/ownership_loss")
    self.scoring_loss = tf.reduce_sum(self.target_weight_used * self.scoring_loss_unreduced, name="losses/scoring_loss")
    self.futurepos_loss = tf.reduce_sum(self.target_weight_used * self.futurepos_loss_unreduced, name="losses/futurepos_loss")
    self.seki_loss = tf.reduce_sum(self.target_weight_used * self.seki_loss_unreduced, name="losses/seki_loss")
    self.scorestdev_reg_loss = tf.reduce_sum(self.target_weight_used * self.scorestdev_reg_loss_unreduced, name="losses/scorestdev_reg_loss")
    # self.winloss_reg_loss = tf.reduce_sum(self.target_weight_used * self.winloss_reg_loss_unreduced, name="losses/winloss_reg_loss")
    self.scale_reg_loss = tf.reduce_sum(self.target_weight_used * self.scale_reg_loss_unreduced, name="losses/scale_reg_loss")

    self.weight_sum = tf.reduce_sum(self.target_weight_used, name="losses/weight_sum")

    if for_optimization:
      #Prior/Regularization
      self.l2_reg_coeff = (placeholders["l2_reg_coeff"] if "l2_reg_coeff" in placeholders else
                           tf.placeholder(tf.float32))
      self.reg_loss_per_weight = self.l2_reg_coeff * (
        tf.add_n([tf.nn.l2_loss(variable) for variable in model.reg_variables]) +
        0.05 * tf.add_n([tf.nn.l2_loss(variable) for variable in model.reg_variables_tiny])
      )
      self.reg_loss = self.reg_loss_per_weight * self.weight_sum

      #The loss to optimize
      self.opt_loss = (
        self.policy_loss +
        self.policy1_loss +
        self.value_loss +
        self.td_value_loss +
        self.scoremean_loss +
        self.lead_loss +
        self.variance_time_loss +
        self.scorebelief_pdf_loss +
        self.scorebelief_cdf_loss +
        self.ownership_loss +
        self.scoring_loss +
        self.futurepos_loss +
        self.seki_loss +
        self.scorestdev_reg_loss +
        self.reg_loss +
        self.scale_reg_loss
      )

      # self.opt_loss = tf.Print(
      #   self.opt_loss,
      #   [self.value_target[0:10],
      #    self.scorevalue_target[0:10],
      #    self.scorebelief_target[0] * tf.constant(model.score_belief_offset_vector,dtype=tf.float32),
      #    self.ownership_target_weight[0:10],
      #    self.selfkomi[0:10],
      #   ],
      #   summarize=2000
      # )

class Metrics:
  def __init__(self,model,target_vars,include_debug_stats):
    #Training results
    policy_target_idxs = tf.argmax(target_vars.policy_target, 1)
    self.top1_prediction = tf.equal(tf.argmax(model.policy_output[:,:,0], 1), policy_target_idxs)
    self.top4_prediction = tf.nn.in_top_k(model.policy_output[:,:,0],policy_target_idxs,4)
    self.accuracy1_unreduced = tf.cast(self.top1_prediction, tf.float32)
    self.accuracy4_unreduced = tf.cast(self.top4_prediction, tf.float32)
    self.value_entropy_unreduced = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(model.value_output,axis=1), logits=model.value_output)
    self.value_conf_unreduced = 4 * tf.square(tf.nn.sigmoid(model.value_output[:,0] - model.value_output[:,1]) - 0.5)
    self.policy_target_entropy_unreduced = target_vars.policy_target_weight * (
      -tf.reduce_sum(target_vars.policy_target * tf.log(target_vars.policy_target+(1e-20)), axis=1)
    )
    self.accuracy1 = tf.reduce_sum(target_vars.target_weight_used * self.accuracy1_unreduced, name="metrics/accuracy1")
    self.accuracy4 = tf.reduce_sum(target_vars.target_weight_used * self.accuracy4_unreduced, name="metrics/accuracy4")
    self.value_entropy = tf.reduce_sum(target_vars.target_weight_used * self.value_entropy_unreduced, name="metrics/value_entropy")
    self.value_conf = tf.reduce_sum(target_vars.target_weight_used * self.value_conf_unreduced, name="metrics/value_conf")
    self.policy_target_entropy = tf.reduce_sum(target_vars.target_weight_used * self.policy_target_entropy_unreduced, name="metrics/policy_target_entropy")
    self.gnorm = None

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
        (name,reduce_stdev(layer,axis=[0,1,2])) for (name,layer) in model.outputs_by_layer
      ])
      self.mean_weights_by_var = dict([
        (v.name,tf.reduce_mean(v)) for v in tf.trainable_variables()
      ])
      self.norm_weights_by_var = dict([
        (v.name,reduce_norm(v)) for v in tf.trainable_variables()
      ])

class ModelUtils:
  @staticmethod
  def print_trainable_variables(logf):
    total_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      total_parameters += variable_parameters
      logf("Model variable: %s, %d parameters" % (variable.name,variable_parameters))

    logf("Model: %d total parameters" % total_parameters)

  @staticmethod
  def build_model_from_tfrecords_features(features,mode,print_model,trainlog,model_config,pos_len,num_batches_per_epoch,lr_scale=None):
    trainlog("Building model")

    num_bin_input_features = Model.get_num_bin_input_features(model_config)
    num_global_input_features = Model.get_num_global_input_features(model_config)

    #L2 regularization coefficient
    if model_config["use_fixup"]:
      l2_coeff_value = 0.000006
    else:
      l2_coeff_value = 0.00003

    placeholders = {}

    binchwp = features["binchwp"]
    #Unpack binary data
    bitmasks = tf.reshape(tf.constant([128,64,32,16,8,4,2,1],dtype=tf.uint8),[1,1,1,8])
    binchw = tf.reshape(tf.bitwise.bitwise_and(tf.expand_dims(binchwp,axis=3),bitmasks),[-1,num_bin_input_features,((pos_len*pos_len+7)//8)*8])
    binchw = binchw[:,:,:pos_len*pos_len]
    binhwc = tf.cast(tf.transpose(binchw, [0,2,1]),tf.float32)
    binhwc = tf.math.minimum(binhwc,tf.constant(1.0))

    placeholders["bin_inputs"] = binhwc

    placeholders["global_inputs"] = features["ginc"]
    placeholders["symmetries"] = tf.greater(tf.random_uniform([3],minval=0,maxval=2,dtype=tf.int32),tf.zeros([3],dtype=tf.int32))

    if mode == tf.estimator.ModeKeys.PREDICT:
      model = Model(model_config,pos_len,placeholders,is_training=False)
      return model

    placeholders["include_history"] = features["gtnc"][:,36:41]

    policy_target0 = features["ptncm"][:,0,:]
    policy_target0 = policy_target0 / tf.reduce_sum(policy_target0,axis=1,keepdims=True)
    placeholders["policy_target"] = policy_target0
    placeholders["policy_target_weight"] = features["gtnc"][:,26]
    policy_target1 = features["ptncm"][:,1,:]
    policy_target1 = policy_target1 / tf.reduce_sum(policy_target1,axis=1,keepdims=True)
    placeholders["policy_target1"] = policy_target1
    placeholders["policy_target_weight1"] = features["gtnc"][:,28]

    placeholders["value_target"] = features["gtnc"][:,0:3]
    placeholders["td_value_target"] = tf.stack([features["gtnc"][:,4:7],features["gtnc"][:,8:11]],axis=1)
    placeholders["scoremean_target"] = features["gtnc"][:,3]
    placeholders["lead_target"] = features["gtnc"][:,21]
    placeholders["variance_time_target"] = features["gtnc"][:,22]
    placeholders["scorebelief_target"] = features["sdn"] / 100.0
    placeholders["ownership_target"] = features["vtnchw"][:,0]
    placeholders["scoring_target"] = features["vtnchw"][:,4] / 120.0
    placeholders["futurepos_target"] = tf.transpose(features["vtnchw"][:,2:4], [0,2,3,1])
    placeholders["seki_target"] = features["vtnchw"][:,1]

    placeholders["target_weight_from_data"] = features["gtnc"][:,25]
    placeholders["lead_target_weight"] = features["gtnc"][:,29]
    placeholders["ownership_target_weight"] = features["gtnc"][:,27]
    placeholders["scoring_target_weight"] = features["gtnc"][:,34]
    placeholders["futurepos_target_weight"] = features["gtnc"][:,33]

    placeholders["selfkomi"] = features["gtnc"][:,47]
    placeholders["l2_reg_coeff"] = tf.constant(l2_coeff_value,dtype=tf.float32)

    if mode == tf.estimator.ModeKeys.EVAL:
      model = Model(model_config,pos_len,placeholders,is_training=False)

      target_vars = Target_vars(model,for_optimization=True,placeholders=placeholders)
      metrics = Metrics(model,target_vars,include_debug_stats=False)
      return (model,target_vars,metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
      model = Model(model_config,pos_len,placeholders,is_training=True)

      target_vars = Target_vars(model,for_optimization=True,placeholders=placeholders)
      metrics = Metrics(model,target_vars,include_debug_stats=False)
      global_step = tf.train.get_global_step()
      global_step_float = tf.cast(global_step, tf.float32)
      global_epoch = global_step_float / tf.constant(num_batches_per_epoch,dtype=tf.float32)

      lr_base = 0.00003 * (1.0 if lr_scale is None else lr_scale)
      per_sample_learning_rate = (
        tf.constant(lr_base) * tf.train.piecewise_constant(
          global_epoch,
          boundaries = [5.0],
          values = [1.0/3.0, 1.0]
        )
      )

      lr_adjusted_variables = model.lr_adjusted_variables
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #collect batch norm update operations
      with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(per_sample_learning_rate, momentum=0.9, use_nesterov=True)
        gradients = optimizer.compute_gradients(target_vars.opt_loss)
        adjusted_gradients = []
        for (grad,x) in gradients:
          adjusted_grad = grad
          if x.name in lr_adjusted_variables and grad is not None:
            adj_factor = lr_adjusted_variables[x.name]
            adjusted_grad = grad * adj_factor
            if print_model:
              trainlog("Adjusting gradient for " + x.name + " by " + str(adj_factor))

          adjusted_gradients.append((adjusted_grad,x))

        gnorm_cap = 2500.0
        (adjusted_gradients_clipped,gnorm) = tf.clip_by_global_norm([x[0] for x in adjusted_gradients],gnorm_cap)
        adjusted_gradients_clipped = list(zip(adjusted_gradients_clipped,[x[1] for x in adjusted_gradients]))
        metrics.gnorm = gnorm
        metrics.excess_gnorm = tf.nn.relu(gnorm-gnorm_cap)
        train_step = optimizer.apply_gradients(adjusted_gradients_clipped, global_step=global_step)

      if print_model:
        ModelUtils.print_trainable_variables(trainlog)
        for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
          trainlog("Additional update op on train step: %s" % update_op.name)
        trainlog("Supporting japanese rules: " + str(model.support_japanese_rules))

      return (model,target_vars,metrics,global_step,global_step_float,per_sample_learning_rate,train_step)
