import tensorflow as tf

from model import Model

# Construct a dictionary that tensorflow uses to know how to parse a tfrecord
def make_raw_input_features(model_config,pos_len,batch_size):
  num_bin_input_features = Model.get_num_bin_input_features(model_config)
  num_global_input_features = Model.get_num_global_input_features(model_config)

  return {
    "binchwp": tf.FixedLenFeature([],tf.string),
    "ginc": tf.FixedLenFeature([batch_size*num_global_input_features],tf.float32),
    "ptncm": tf.FixedLenFeature([batch_size*Model.NUM_POLICY_TARGETS*(pos_len*pos_len+1)],tf.float32),
    "gtnc": tf.FixedLenFeature([batch_size*Model.NUM_GLOBAL_TARGETS],tf.float32),
    "sdn": tf.FixedLenFeature([batch_size*(pos_len*pos_len*2+Model.EXTRA_SCORE_DISTR_RADIUS*2)],tf.float32),
    "sbsn": tf.FixedLenFeature([batch_size*(Model.BONUS_SCORE_RADIUS*2+1)],tf.float32),
    "vtnchw": tf.FixedLenFeature([batch_size*Model.NUM_VALUE_SPATIAL_TARGETS*pos_len*pos_len],tf.float32)
  }

# Construct a dictionary of placeholders, in case we're using a feed_dict_like way of providing
# training rows rather than via dataset
def make_raw_input_feature_placeholders(model_config,pos_len,batch_size):
  num_bin_input_features = Model.get_num_bin_input_features(model_config)
  num_global_input_features = Model.get_num_global_input_features(model_config)

  return {
    "binchwp": tf.placeholder(tf.uint8,[batch_size,num_bin_input_features,(pos_len*pos_len+7)//8]),
    "ginc": tf.placeholder(tf.float32,[batch_size,num_global_input_features]),
    "ptncm": tf.placeholder(tf.float32,[batch_size,Model.NUM_POLICY_TARGETS,pos_len*pos_len+1]),
    "gtnc": tf.placeholder(tf.float32,[batch_size,Model.NUM_GLOBAL_TARGETS]),
    "sdn": tf.placeholder(tf.float32,[batch_size,pos_len*pos_len*2+Model.EXTRA_SCORE_DISTR_RADIUS*2]),
    "sbsn": tf.placeholder(tf.float32,[batch_size,Model.BONUS_SCORE_RADIUS*2+1]),
    "vtnchw": tf.placeholder(tf.float32,[batch_size,Model.NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
  }

# Return a function for parsing a tfrecord
# (or rather, the function that transforms an input pipe of tfrecords into a tensor for the outputs
# to go once the dataset begins running)
def make_tf_record_parser(model_config,pos_len,batch_size):
  num_bin_input_features = Model.get_num_bin_input_features(model_config)
  num_global_input_features = Model.get_num_global_input_features(model_config)
  raw_input_features = make_raw_input_features(model_config,pos_len,batch_size)

  def parse_input(serialized_example):
    example = tf.parse_single_example(serialized_example,raw_input_features)
    binchwp = tf.decode_raw(example["binchwp"],tf.uint8)
    ginc = example["ginc"]
    ptncm = example["ptncm"]
    gtnc = example["gtnc"]
    sdn = example["sdn"]
    sbsn = example["sbsn"]
    vtnchw = example["vtnchw"]
    return {
      "binchwp": tf.reshape(binchwp,[batch_size,num_bin_input_features,(pos_len*pos_len+7)//8]),
      "ginc": tf.reshape(ginc,[batch_size,num_global_input_features]),
      "ptncm": tf.reshape(ptncm,[batch_size,Model.NUM_POLICY_TARGETS,pos_len*pos_len+1]),
      "gtnc": tf.reshape(gtnc,[batch_size,Model.NUM_GLOBAL_TARGETS]),
      "sdn": tf.reshape(sdn,[batch_size,pos_len*pos_len*2+Model.EXTRA_SCORE_DISTR_RADIUS*2]),
      "sbsn": tf.reshape(sbsn,[batch_size,Model.BONUS_SCORE_RADIUS*2+1]),
      "vtnchw": tf.reshape(vtnchw,[batch_size,Model.NUM_VALUE_SPATIAL_TARGETS,pos_len,pos_len])
    }

  return parse_input

# Create a tf.train.Example from the given start:stop interval of numpy arrays of data.
# Used in shuffler to output tfrecords

def make_tf_record_example(
    binaryInputNCHWPacked,
    globalInputNC,
    policyTargetsNCMove,
    globalTargetsNC,
    scoreDistrN,
    selfBonusScoreN,
    valueTargetsNCHW,
    start,
    stop
):
  example = tf.train.Example(features=tf.train.Features(feature={
    "binchwp": tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[binaryInputNCHWPacked[start:stop].reshape(-1).tobytes()])
    ),
    "ginc": tf.train.Feature(
      float_list=tf.train.FloatList(value=globalInputNC[start:stop].reshape(-1))
    ),
    "ptncm": tf.train.Feature(
      float_list=tf.train.FloatList(value=policyTargetsNCMove[start:stop].reshape(-1))
    ),
    "gtnc": tf.train.Feature(
      float_list=tf.train.FloatList(value=globalTargetsNC[start:stop].reshape(-1))
    ),
    "sdn": tf.train.Feature(
      float_list=tf.train.FloatList(value=scoreDistrN[start:stop].reshape(-1))
    ),
    "sbsn": tf.train.Feature(
      float_list=tf.train.FloatList(value=selfBonusScoreN[start:stop].reshape(-1))
    ),
    "vtnchw": tf.train.Feature(
      float_list=tf.train.FloatList(value=valueTargetsNCHW[start:stop].reshape(-1))
    )
  }))
  return example
