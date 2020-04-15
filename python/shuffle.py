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
from tensorflow.python_io import TFRecordOptions,TFRecordCompressionType,TFRecordWriter

import tfrecordio

keys = [
  "binaryInputNCHWPacked",
  "globalInputNC",
  "policyTargetsNCMove",
  "globalTargetsNC",
  "scoreDistrN",
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

  #print("Shardify reading: " + input_file)
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
  valueTargetsNCHW = npz["valueTargetsNCHW"]

  joint_shuffle((binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW))

  num_rows_to_keep = binaryInputNCHWPacked.shape[0]
  assert(globalInputNC.shape[0] == num_rows_to_keep)
  assert(policyTargetsNCMove.shape[0] == num_rows_to_keep)
  assert(globalTargetsNC.shape[0] == num_rows_to_keep)
  assert(scoreDistrN.shape[0] == num_rows_to_keep)
  assert(valueTargetsNCHW.shape[0] == num_rows_to_keep)

  if keep_prob < 1.0:
    num_rows_to_keep = min(num_rows_to_keep,int(round(num_rows_to_keep * keep_prob)))

  rand_assts = np.random.randint(num_out_files,size=[num_rows_to_keep])
  counts = np.bincount(rand_assts,minlength=num_out_files)
  countsums = np.cumsum(counts)
  assert(countsums[len(countsums)-1] == num_rows_to_keep)

  #print("Shardify writing... (mem usage %dMB)" % memusage_mb())
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
      valueTargetsNCHW = valueTargetsNCHW[start:stop]
    )
  return num_out_files

def merge_shards(filename, num_shards_to_merge, out_tmp_dir, batch_size, ensure_batch_multiple):
  tfoptions = TFRecordOptions(TFRecordCompressionType.ZLIB)
  record_writer = TFRecordWriter(filename,tfoptions)

  binaryInputNCHWPackeds = []
  globalInputNCs = []
  policyTargetsNCMoves = []
  globalTargetsNCs = []
  scoreDistrNs = []
  valueTargetsNCHWs = []

  for input_idx in range(num_shards_to_merge):
    shard_filename = os.path.join(out_tmp_dir, str(input_idx) + ".npz")
    npz = np.load(shard_filename)
    assert(set(npz.keys()) == set(keys))

    binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
    globalInputNC = npz["globalInputNC"]
    policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
    globalTargetsNC = npz["globalTargetsNC"]
    scoreDistrN = npz["scoreDistrN"].astype(np.float32)
    valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)

    binaryInputNCHWPackeds.append(binaryInputNCHWPacked)
    globalInputNCs.append(globalInputNC)
    policyTargetsNCMoves.append(policyTargetsNCMove)
    globalTargetsNCs.append(globalTargetsNC)
    scoreDistrNs.append(scoreDistrN)
    valueTargetsNCHWs.append(valueTargetsNCHW)

  ###
  #WARNING - if adding anything here, also add it to joint_shuffle below!
  ###
  binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackeds)
  globalInputNC = np.concatenate(globalInputNCs)
  policyTargetsNCMove = np.concatenate(policyTargetsNCMoves)
  globalTargetsNC = np.concatenate(globalTargetsNCs)
  scoreDistrN = np.concatenate(scoreDistrNs)
  valueTargetsNCHW = np.concatenate(valueTargetsNCHWs)

  joint_shuffle((binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW))

  num_rows = binaryInputNCHWPacked.shape[0]
  #Just truncate and lose the batch at the end, it's fine
  num_batches = (num_rows // (batch_size * ensure_batch_multiple)) * ensure_batch_multiple
  for i in range(num_batches):
    start = i*batch_size
    stop = (i+1)*batch_size

    example = tfrecordio.make_tf_record_example(
      binaryInputNCHWPacked,
      globalInputNC,
      policyTargetsNCMove,
      globalTargetsNC,
      scoreDistrN,
      valueTargetsNCHW,
      start,
      stop
    )
    record_writer.write(example.SerializeToString())

  jsonfilename = os.path.splitext(filename)[0] + ".json"
  with open(jsonfilename,"w") as f:
    json.dump({"num_rows":num_rows,"num_batches":num_batches},f)

  record_writer.close()
  return num_batches * batch_size


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Shuffle data files')
  parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')
  parser.add_argument('-min-rows', type=int, required=False, help='Minimum training rows to use, default 250k')
  parser.add_argument('-max-rows', type=int, required=False, help='Maximum training rows to use, default unbounded')
  parser.add_argument('-keep-target-rows', type=int, required=False, help='Target number of rows to actually keep in the final data set, default 1.2M')
  parser.add_argument('-expand-window-per-row', type=float, required=True, help='Beyond min rows, initially expand the window by this much every post-random data row')
  parser.add_argument('-taper-window-exponent', type=float, required=True, help='Make the window size asymtotically grow as this power of the data rows')
  parser.add_argument('-out-dir', required=True, help='Dir to output training files')
  parser.add_argument('-out-tmp-dir', required=True, help='Dir to use as scratch space')
  parser.add_argument('-approx-rows-per-out-file', type=int, required=True, help='Number of rows per output tf records file')
  parser.add_argument('-num-processes', type=int, required=True, help='Number of multiprocessing processes')
  parser.add_argument('-batch-size', type=int, required=True, help='Batch size to write training examples in')
  parser.add_argument('-ensure-batch-multiple', type=int, required=False, help='Ensure each file is a multiple of this many batches')

  args = parser.parse_args()
  dirs = args.dirs
  min_rows = args.min_rows
  max_rows = args.max_rows
  keep_target_rows = args.keep_target_rows
  expand_window_per_row = args.expand_window_per_row
  taper_window_exponent = args.taper_window_exponent
  out_dir = args.out_dir
  out_tmp_dir = args.out_tmp_dir
  approx_rows_per_out_file = args.approx_rows_per_out_file
  num_processes = args.num_processes
  batch_size = args.batch_size
  ensure_batch_multiple = 1
  if args.ensure_batch_multiple is not None:
    ensure_batch_multiple = args.ensure_batch_multiple
  if min_rows is None:
    print("NOTE: -min-rows was not specified, defaulting to requiring 250K rows before shuffling.")
    min_rows = 250000
  if keep_target_rows is None:
    print("NOTE: -keep-target-rows was not specified, defaulting to keeping the first 1.2M rows.")
    print("(slightly larger than default training epoch size of 1M, to give 1 epoch of data regardless of discreteness rows or batches per output file)")
    print("If you intended to shuffle the whole dataset instead, pass in -keep-target-rows <very large number>")
    keep_target_rows = 1200000

  all_files = []
  for d in dirs:
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
  num_rows_total = 0 #Number of data rows
  num_random_rows_capped = 0 #Number of random data rows, capped at min_rows - we never keep more than min_rows many data rows if they're from random.
  num_postrandom_rows = 0 #Number of NON-random rows

  #How far offset do we start on the power-law window tail? E.g. what number of postrandom rows do we need before the window size grows by a factor
  #of 2^(taper_window_exponent)? For now, we set it equal to the min rows
  window_taper_offset = min_rows

  def num_usable_rows():
    global num_random_rows_capped
    global num_postrandom_rows
    return num_random_rows_capped + num_postrandom_rows
  def num_desired_rows():
    #Every post-random row moves one row beyond window_taper_offset
    power_law_x = num_usable_rows() - min_rows + window_taper_offset
    #Apply power law and correct for window_taper_offset so we're still anchored at 0
    unscaled_power_law = (power_law_x ** taper_window_exponent) - (window_taper_offset ** taper_window_exponent)
    #Scale so that we have an initial derivative of 1
    scaled_power_law = unscaled_power_law / (taper_window_exponent * (window_taper_offset ** (taper_window_exponent-1)))
    #Scale so that we have the desired initial slope, and add back the minimum random rows
    return int(scaled_power_law * expand_window_per_row + min_rows)

  for (filename,mtime) in all_files:
    npheaders = get_numpy_npz_headers(filename)
    if npheaders is None or len(npheaders) <= 0:
      continue
    (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
    num_rows = shape[0]
    row_range = (num_rows_total, num_rows_total + num_rows)
    num_rows_total += num_rows
    if "random" not in filename:
      num_postrandom_rows += num_rows
    else:
      num_random_rows_capped = min(num_random_rows_capped + num_rows, min_rows)

    files_with_row_range.append((filename,row_range))

    #If we already have a window size bigger than max, then just stop
    if max_rows is not None and num_desired_rows() >= max_rows:
      break

  if os.path.exists(out_dir):
    raise Exception(out_dir + " already exists")
  os.mkdir(out_dir)

  if num_rows_total <= 0:
    print("No rows found")
    sys.exit(0)

  #If we don't have enough rows, then quit out
  if num_rows_total < min_rows:
    print("Not enough rows, only %d (fewer than %d)" % (num_rows_total,min_rows))
    sys.exit(0)

  print("Total rows found: %d (%d usable)" % (num_rows_total,num_usable_rows()))

  #Reverse so that recent files are first
  files_with_row_range.reverse()

  #Now assemble only the files we need to hit our desired window size
  desired_num_rows = num_desired_rows()
  desired_num_rows = max(desired_num_rows,min_rows)
  desired_num_rows = min(desired_num_rows,max_rows) if max_rows is not None else desired_num_rows
  print("Desired num rows: %d / %d" % (desired_num_rows,num_rows_total))

  desired_input_files = []
  desired_input_files_with_row_range = []
  num_rows_total = 0
  len_files_with_row_range = len(files_with_row_range)
  print_stride = 1 + len(files_with_row_range) // 40
  for i in range(len(files_with_row_range)):
    (filename,(start_row,end_row)) = files_with_row_range[i]
    desired_input_files.append(filename)
    desired_input_files_with_row_range.append((filename,(start_row,end_row)))

    num_rows_total += (end_row - start_row)
    if i % print_stride == 0 or num_rows_total >= desired_num_rows or i == len_files_with_row_range - 1:
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
    print("Beginning sharding",flush=True)
    t0 = time.time()
    shard_results = pool.starmap(shardify, [
      (input_idx, desired_input_files[input_idx], num_out_files, out_tmp_dirs, keep_prob) for input_idx in range(len(desired_input_files))
    ])
    t1 = time.time()
    print("Done sharding, number of shards by input file:",flush=True)
    # print(list(zip(desired_input_files,shard_results)),flush=True)
    print("Time taken: " + str(t1-t0),flush=True)
    sys.stdout.flush()

    print("Beginning merging",flush=True)
    t0 = time.time()
    num_shards_to_merge = len(desired_input_files)
    merge_results = pool.starmap(merge_shards, [
      (out_files[idx],num_shards_to_merge,out_tmp_dirs[idx],batch_size,ensure_batch_multiple) for idx in range(len(out_files))
    ])
    t1 = time.time()
    print("Done merging, number of rows by output file:",flush=True)
    print(list(zip(out_files,merge_results)),flush=True)
    print("Time taken: " + str(t1-t0),flush=True)
    sys.stdout.flush()

  clean_tmp_dirs()

  dump_value = {
    "files": files_with_row_range,
    "range": (min(start_row for (filename,(start_row,end_row)) in desired_input_files_with_row_range),
              max(end_row for (filename,(start_row,end_row)) in desired_input_files_with_row_range))
  }

  with open(out_dir + ".json", 'w') as f:
    json.dump(dump_value, f)
