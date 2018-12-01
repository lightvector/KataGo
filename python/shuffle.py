#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import math
import time
import logging
import zipfile
import shutil
import psutil
import json

import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.python_io import *

keys = [
  "binaryInputNCHWPacked",
  "globalInputNC",
  "policyTargetsNCMove",
  "globalTargetsNC",
  "scoreDistrN",
  "selfBonusScoreN",
  "valueTargetsNCHW"
]

def joint_shuffle(arrs):
  rand_state = np.random.get_state()
  for arr in arrs:
    assert(len(arr) == len(arrs[0]))
  for arr in arrs:
    np.random.set_state(rand_state)
    np.random.shuffle(arr)

def memusage_mb():
  return psutil.Process(os.getpid()).memory_info().rss // 1048576

def shardify(input_idx, input_file, num_out_files, out_tmp_dirs, keep_prob):
  np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(4)])

  print("Reading: " + input_file)
  npz = np.load(input_file)
  assert(set(npz.keys()) == set(keys))

  ###
  #WARNING - if adding anything here, also add it to joint_shuffle below!
  ###
  binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
  globalInputNC = npz["globalInputNC"]
  policyTargetsNCMove = npz["policyTargetsNCMove"]
  globalTargetsNC = npz["globalTargetsNC"]
  scoreDistrN = npz["scoreDistrN"]
  selfBonusScoreN = npz["selfBonusScoreN"]
  valueTargetsNCHW = npz["valueTargetsNCHW"]

  print("Shuffling... (mem usage %dMB)" % memusage_mb())
  joint_shuffle((binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,selfBonusScoreN,valueTargetsNCHW))

  num_rows_to_keep = binaryInputNCHWPacked.shape[0]
  assert(globalInputNC.shape[0] == num_rows_to_keep)
  assert(policyTargetsNCMove.shape[0] == num_rows_to_keep)
  assert(globalTargetsNC.shape[0] == num_rows_to_keep)
  assert(scoreDistrN.shape[0] == num_rows_to_keep)
  assert(selfBonusScoreN.shape[0] == num_rows_to_keep)
  assert(valueTargetsNCHW.shape[0] == num_rows_to_keep)

  if keep_prob < 1.0:
    num_rows_to_keep = min(num_rows_to_keep,int(round(num_rows_to_keep * keep_prob)))

  rand_assts = np.random.randint(num_out_files,size=[num_rows_to_keep])
  counts = np.bincount(rand_assts,minlength=num_out_files)
  countsums = np.cumsum(counts)
  assert(countsums[len(countsums)-1] == num_rows_to_keep)

  print("Writing shards... (mem usage %dMB)" % memusage_mb())
  for out_idx in range(num_out_files):
    start = countsums[out_idx]-counts[out_idx]
    stop = countsums[out_idx]
    np.savez_compressed(
      os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
      binaryInputNCHWPacked = binaryInputNCHWPacked[start:stop],
      globalInputNC = globalInputNC[start:stop],
      policyTargetsNCMove = policyTargetsNCMove[start:stop],
      globalTargetsNC = globalTargetsNC[start:stop],
      scoreDistrN = scoreDistrN[start:stop],
      selfBonusScoreN = selfBonusScoreN[start:stop],
      valueTargetsNCHW = valueTargetsNCHW[start:stop]
    )

def merge_shards(filename, num_shards_to_merge, out_tmp_dir, batch_size):
  print("Merging shards for output file: %s (%d shards to merge)" % (filename,num_shards_to_merge))
  tfoptions = TFRecordOptions(TFRecordCompressionType.ZLIB)
  record_writer = TFRecordWriter(filename,tfoptions)

  binaryInputNCHWPackeds = []
  globalInputNCs = []
  policyTargetsNCMoves = []
  globalTargetsNCs = []
  scoreDistrNs = []
  selfBonusScoreNs = []
  valueTargetsNCHWs = []

  for input_idx in range(num_shards_to_merge):
    shard_filename = os.path.join(out_tmp_dir, str(input_idx) + ".npz")
    print("Loading shard: %d (mem usage %dMB)" % (input_idx,memusage_mb()))

    npz = np.load(shard_filename)
    assert(set(npz.keys()) == set(keys))

    binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
    globalInputNC = npz["globalInputNC"]
    policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
    globalTargetsNC = npz["globalTargetsNC"]
    scoreDistrN = npz["scoreDistrN"].astype(np.float32)
    selfBonusScoreN = npz["selfBonusScoreN"].astype(np.float32)
    valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)

    binaryInputNCHWPackeds.append(binaryInputNCHWPacked)
    globalInputNCs.append(globalInputNC)
    policyTargetsNCMoves.append(policyTargetsNCMove)
    globalTargetsNCs.append(globalTargetsNC)
    scoreDistrNs.append(scoreDistrN)
    selfBonusScoreNs.append(selfBonusScoreN)
    valueTargetsNCHWs.append(valueTargetsNCHW)

  ###
  #WARNING - if adding anything here, also add it to joint_shuffle below!
  ###
  print("Concatenating... (mem usage %dMB)" % memusage_mb())
  binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackeds)
  globalInputNC = np.concatenate(globalInputNCs)
  policyTargetsNCMove = np.concatenate(policyTargetsNCMoves)
  globalTargetsNC = np.concatenate(globalTargetsNCs)
  scoreDistrN = np.concatenate(scoreDistrNs)
  selfBonusScoreN = np.concatenate(selfBonusScoreNs)
  valueTargetsNCHW = np.concatenate(valueTargetsNCHWs)

  print("Shuffling... (mem usage %dMB)" % memusage_mb())
  joint_shuffle((binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,selfBonusScoreN,valueTargetsNCHW))

  print("Writing in batches...")
  num_rows = binaryInputNCHWPacked.shape[0]
  #Just truncate and lose the batch at the end, it's fine
  num_batches = num_rows // batch_size
  for i in range(num_batches):
    start = i*batch_size
    stop = (i+1)*batch_size
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
    record_writer.write(example.SerializeToString())

  jsonfilename = os.path.splitext(filename)[0] + ".json"
  with open(jsonfilename,"w") as f:
    json.dump({"num_rows":num_rows,"num_batches":num_batches},f)

  print("Done %s (%d rows)" % (filename, num_batches * batch_size))

  record_writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Shuffle data files')
  parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')
  parser.add_argument('-min-rows', type=int, required=True, help='Minimum training rows to use')
  parser.add_argument('-max-rows', type=int, required=True, help='Maximum training rows to use')
  parser.add_argument('-keep-target-rows', type=int, required=False, help='Target number of rows to actually keep in the final data set')
  parser.add_argument('-window-factor', type=float, required=True, help='Beyond min rows, add 1 more row per this many')
  parser.add_argument('-out-dir', required=True, help='Dir to output training files')
  parser.add_argument('-out-tmp-dir', required=True, help='Dir to use as scratch space')
  parser.add_argument('-approx-rows-per-out-file', type=int, required=True, help='Number of rows per output tf records file')
  parser.add_argument('-num-processes', type=int, required=True, help='Number of multiprocessing processes')
  parser.add_argument('-batch-size', type=int, required=True, help='Batck size to write training examples in')

  args = parser.parse_args()
  dirs = args.dirs
  min_rows = args.min_rows
  max_rows = args.max_rows
  keep_target_rows = args.keep_target_rows
  window_factor = args.window_factor
  out_dir = args.out_dir
  out_tmp_dir = args.out_tmp_dir
  approx_rows_per_out_file = args.approx_rows_per_out_file
  num_processes = args.num_processes
  batch_size = args.batch_size

  all_files = []
  for d in dirs:
    print(d)
    for (path,dirnames,filenames) in os.walk(d):
      filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
      filenames = [(filename,os.path.getmtime(filename)) for filename in filenames]
      all_files.extend(filenames)

  all_files.sort(key=(lambda x: x[1]), reverse=False)

  def get_numpy_npz_headers(filename):
    with zipfile.ZipFile(filename) as z:
      wasbad = False
      numrows = 0
      npzheaders = {}
      for subfilename in z.namelist():
        npyfile = z.open(subfilename)
        try:
          version = np.lib.format.read_magic(npyfile)
        except ValueError:
          wasbad = True
          print("WARNING: bad file, skipping it: %s (bad array %s)" % (filename,subfilename))
        else:
          (shape, is_fortran, dtype) = np.lib.format._read_array_header(npyfile,version)
          npzheaders[subfilename] = (shape, is_fortran, dtype)
      if wasbad:
        return None
      return npzheaders


  files_with_row_range = []
  num_rows_total = 0
  for (filename,mtime) in all_files:
    npheaders = get_numpy_npz_headers(filename)
    if npheaders is None or len(npheaders) <= 0:
      continue
    (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
    num_rows = shape[0]
    row_range = (num_rows_total, num_rows_total + num_rows)
    num_rows_total += num_rows

    print("Training data file %s: %d rows" % (filename,num_rows))
    files_with_row_range.append((filename,row_range))

    #If we have more rows than we could possibly need to hit max rows, then just stop
    if num_rows_total >= min_rows + (max_rows - min_rows) * window_factor:
      break

  if os.path.exists(out_dir):
    raise Exception(out_dir + " already exists")
  os.mkdir(out_dir)

  if num_rows_total <= 0:
    print("No rows found")
    sys.exit(0)

  #If we don't have enough rows, then quit out
  if num_rows_total < min_rows:
    print("Not enough rows (fewer than %d)" % min_rows)
    sys.exit(0)

  print("Total rows found: %d" % num_rows_total)

  #Reverse so that recent files are first
  files_with_row_range.reverse()

  #Now assemble only the files we need to hit our desired window size
  desired_num_rows = int(min_rows + (num_rows_total - min_rows) / window_factor)
  desired_num_rows = max(desired_num_rows,min_rows)
  desired_num_rows = min(desired_num_rows,max_rows)
  print("Desired num rows: %d" % desired_num_rows)

  desired_input_files = []
  desired_input_files_with_row_range = []
  num_rows_total = 0
  for (filename,(start_row,end_row)) in files_with_row_range:
    desired_input_files.append(filename)
    desired_input_files_with_row_range.append((filename,(start_row,end_row)))

    num_rows_total += (end_row - start_row)
    print("Using: %s (%d-%d) (%d/%d desired rows)" % (filename,start_row,end_row,num_rows_total,desired_num_rows))
    if num_rows_total >= desired_num_rows:
      break

  np.random.seed()
  np.random.shuffle(desired_input_files)

  approx_rows_to_keep = num_rows_total
  if keep_target_rows is not None:
    approx_rows_to_keep = min(approx_rows_to_keep, keep_target_rows)
  keep_prob = approx_rows_to_keep / num_rows_total

  num_out_files = int(round(approx_rows_to_keep / approx_rows_per_out_file))
  num_out_files = max(num_out_files,1)

  out_files = [os.path.join(out_dir, "data%d.tfrecord" % i) for i in range(num_out_files)]
  out_tmp_dirs = [os.path.join(out_tmp_dir, "tmp.shuf%d" % i) for i in range(num_out_files)]
  print("Writing %d output files" % num_out_files)

  def clean_tmp_dirs():
    for tmp_dir in out_tmp_dirs:
      if os.path.exists(tmp_dir):
        print("Cleaning up tmp dir: " + tmp_dir)
        shutil.rmtree(tmp_dir)

  clean_tmp_dirs()
  for tmp_dir in out_tmp_dirs:
    os.mkdir(tmp_dir)

  with multiprocessing.Pool(num_processes) as pool:
    pool.starmap(shardify, [
      (input_idx, desired_input_files[input_idx], num_out_files, out_tmp_dirs, keep_prob) for input_idx in range(len(desired_input_files))
    ])

    num_shards_to_merge = len(desired_input_files)
    pool.starmap(merge_shards, [
      (out_files[idx],num_shards_to_merge,out_tmp_dirs[idx],batch_size) for idx in range(len(out_files))
    ])

  clean_tmp_dirs()
  with open(out_dir + ".json", 'w') as f:
    json.dump(desired_input_files_with_row_range, f)
