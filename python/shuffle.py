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

def joint_shuffle_take_first_n(n,arrs):
  for arr in arrs:
    assert(len(arr) == len(arrs[0]))
  perm = np.random.permutation(len(arrs[0]))
  perm = perm[:n]
  shuffled_arrs = []
  for arr in arrs:
    shuffled_arrs.append(arr[perm])
  return shuffled_arrs

def memusage_mb():
  return psutil.Process(os.getpid()).memory_info().rss // 1048576

def shardify(input_idx, input_file_group, num_out_files, out_tmp_dirs, keep_prob):
  np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(4)])

  assert(len(input_file_group) > 0)

  if len(input_file_group) == 1:
    with np.load(input_file_group[0]) as npz:
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
  else:
    binaryInputNCHWPackedList = []
    globalInputNCList = []
    policyTargetsNCMoveList = []
    globalTargetsNCList = []
    scoreDistrNList = []
    valueTargetsNCHWList = []

    for input_file in input_file_group:
      with np.load(input_file) as npz:
        assert(set(npz.keys()) == set(keys))
        binaryInputNCHWPackedList.append(npz["binaryInputNCHWPacked"])
        globalInputNCList.append(npz["globalInputNC"])
        policyTargetsNCMoveList.append(npz["policyTargetsNCMove"])
        globalTargetsNCList.append(npz["globalTargetsNC"])
        scoreDistrNList.append(npz["scoreDistrN"])
        valueTargetsNCHWList.append(npz["valueTargetsNCHW"])

    binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackedList,axis=0)
    globalInputNC = np.concatenate(globalInputNCList,axis=0)
    policyTargetsNCMove = np.concatenate(policyTargetsNCMoveList,axis=0)
    globalTargetsNC = np.concatenate(globalTargetsNCList,axis=0)
    scoreDistrN = np.concatenate(scoreDistrNList,axis=0)
    valueTargetsNCHW = np.concatenate(valueTargetsNCHWList,axis=0)

  num_rows_to_keep = binaryInputNCHWPacked.shape[0]
  assert(globalInputNC.shape[0] == num_rows_to_keep)
  assert(policyTargetsNCMove.shape[0] == num_rows_to_keep)
  assert(globalTargetsNC.shape[0] == num_rows_to_keep)
  assert(scoreDistrN.shape[0] == num_rows_to_keep)
  assert(valueTargetsNCHW.shape[0] == num_rows_to_keep)

  if keep_prob < 1.0:
    num_rows_to_keep = min(num_rows_to_keep,int(round(num_rows_to_keep * keep_prob)))

  [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW] = (
    joint_shuffle_take_first_n(num_rows_to_keep,[binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW])
  )

  assert(binaryInputNCHWPacked.shape[0] == num_rows_to_keep)
  assert(globalInputNC.shape[0] == num_rows_to_keep)
  assert(policyTargetsNCMove.shape[0] == num_rows_to_keep)
  assert(globalTargetsNC.shape[0] == num_rows_to_keep)
  assert(scoreDistrN.shape[0] == num_rows_to_keep)
  assert(valueTargetsNCHW.shape[0] == num_rows_to_keep)

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
  np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(5)])

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
    with np.load(shard_filename) as npz:
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

  num_rows = binaryInputNCHWPacked.shape[0]
  assert(globalInputNC.shape[0] == num_rows)
  assert(policyTargetsNCMove.shape[0] == num_rows)
  assert(globalTargetsNC.shape[0] == num_rows)
  assert(scoreDistrN.shape[0] == num_rows)
  assert(valueTargetsNCHW.shape[0] == num_rows)

  [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW] = (
    joint_shuffle_take_first_n(num_rows,[binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW])
  )

  assert(binaryInputNCHWPacked.shape[0] == num_rows)
  assert(globalInputNC.shape[0] == num_rows)
  assert(policyTargetsNCMove.shape[0] == num_rows)
  assert(globalTargetsNC.shape[0] == num_rows)
  assert(scoreDistrN.shape[0] == num_rows)
  assert(valueTargetsNCHW.shape[0] == num_rows)

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


def compute_num_rows(filename):
  npheaders = get_numpy_npz_headers(filename)
  if npheaders is None or len(npheaders) <= 0:
    return (filename,0)
  (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
  num_rows = shape[0]
  return (filename,num_rows)


class TimeStuff(object):

  def __init__(self,taskstr):
    self.taskstr = taskstr

  def __enter__(self):
    print("Beginning: %s" % self.taskstr, flush=True)
    self.t0 = time.time()

  def __exit__(self, exception_type, exception_val, trace):
    self.t1 = time.time()
    print("Finished: %s in %s seconds" % (self.taskstr, str(self.t1 - self.t0)), flush=True)
    return True


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Shuffle data files')
  parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')
  parser.add_argument('-min-rows', type=int, required=False, help='Minimum training rows to use, default 250k')
  parser.add_argument('-max-rows', type=int, required=False, help='Maximum training rows to use, default unbounded')
  parser.add_argument('-keep-target-rows', type=int, required=False, help='Target number of rows to actually keep in the final data set, default 1.2M')
  parser.add_argument('-expand-window-per-row', type=float, required=True, help='Beyond min rows, initially expand the window by this much every post-random data row')
  parser.add_argument('-taper-window-exponent', type=float, required=True, help='Make the window size asymtotically grow as this power of the data rows')
  parser.add_argument('-taper-window-scale', type=float, required=False, help='The scale at which the power law applies')
  parser.add_argument('-add-to-window', type=float, required=False, help='Compute as if the window size were this much larger/smaller')
  parser.add_argument('-summary-file', required=False, help='Summary json file for directory contents')
  parser.add_argument('-out-dir', required=True, help='Dir to output training files')
  parser.add_argument('-out-tmp-dir', required=True, help='Dir to use as scratch space')
  parser.add_argument('-approx-rows-per-out-file', type=int, required=True, help='Number of rows per output tf records file')
  parser.add_argument('-num-processes', type=int, required=True, help='Number of multiprocessing processes')
  parser.add_argument('-batch-size', type=int, required=True, help='Batch size to write training examples in')
  parser.add_argument('-ensure-batch-multiple', type=int, required=False, help='Ensure each file is a multiple of this many batches')
  parser.add_argument('-worker-group-size', type=int, required=False, help='Internally, target having many rows per parallel sharding worker')
  parser.add_argument('-exclude', required=False, help='Text file with npzs to ignore, one per line')
  parser.add_argument('-exclude-prefix', required=False, help='Prefix to concat to lines in exclude to produce the full file path')

  args = parser.parse_args()
  dirs = args.dirs
  min_rows = args.min_rows
  max_rows = args.max_rows
  keep_target_rows = args.keep_target_rows
  expand_window_per_row = args.expand_window_per_row
  taper_window_exponent = args.taper_window_exponent
  taper_window_scale = args.taper_window_scale
  add_to_window = args.add_to_window
  summary_file = args.summary_file
  out_dir = args.out_dir
  out_tmp_dir = args.out_tmp_dir
  approx_rows_per_out_file = args.approx_rows_per_out_file
  num_processes = args.num_processes
  batch_size = args.batch_size
  ensure_batch_multiple = 1
  if args.ensure_batch_multiple is not None:
    ensure_batch_multiple = args.ensure_batch_multiple
  worker_group_size = args.worker_group_size
  if worker_group_size is None:
    worker_group_size = 80000
  exclude = args.exclude
  exclude_prefix = args.exclude_prefix
  if exclude_prefix is None:
    exclude_prefix = ""

  if min_rows is None:
    print("NOTE: -min-rows was not specified, defaulting to requiring 250K rows before shuffling.")
    min_rows = 250000
  if keep_target_rows is None:
    print("NOTE: -keep-target-rows was not specified, defaulting to keeping the first 1.5M rows.")
    print("(slightly larger than default training epoch size of 1M, to give 1 epoch of data regardless of discreteness rows or batches per output file)")
    print("If you intended to shuffle the whole dataset instead, pass in -keep-target-rows <very large number>")
    keep_target_rows = 1500000
  if add_to_window is None:
    add_to_window = 0

  summary_data_by_dirpath = {}
  if summary_file is not None:
    with TimeStuff("Loading " + summary_file):
      # Try a bunch of times, just to be robust to if the file is being swapped out in nfs
      for i in range(10):
        success = False
        try:
          with open(summary_file) as fp:
            summary_data_by_dirpath = json.load(fp)
            success = True
        except OSError:
          success = False
        except ValueError:
          success = False
        if success:
          break
        time.sleep(1)
      if not success:
        raise RuntimeError("Could not load summary file")

  exclude_set = set()
  if exclude is not None:
    with TimeStuff("Loading " + exclude):
      # Try a bunch of times, just to be robust to if the file is being swapped out in nfs
      for i in range(10):
        success = False
        try:
          with open(exclude,"r") as exclude_in:
            excludes = exclude_in.readlines()
            excludes = [x.strip() for x in excludes]
            excludes = [x for x in excludes if len(x) > 0]
            excludes = [exclude_prefix + x for x in excludes]
            exclude_set = set(excludes)
            success = True
        except OSError:
          success = False
        except ValueError:
          success = False
        if success:
          break
        time.sleep(1)
      if not success:
        raise RuntimeError("Could not load summary file")


  all_files = []
  files_with_unknown_num_rows = []
  excluded_count = 0
  with TimeStuff("Finding files"):
    for d in dirs:
      for (path,dirnames,filenames) in os.walk(d, followlinks=True):
        i = 0
        while i < len(dirnames):
          dirname = dirnames[i]
          filename_mtime_num_rowss = summary_data_by_dirpath.get(os.path.abspath(os.path.join(path, dirname)))
          if filename_mtime_num_rowss is not None:
            del dirnames[i]
            i -= 1
            for (filename,mtime,num_rows) in filename_mtime_num_rowss:
              filename = os.path.join(path,dirname,filename)
              if filename in exclude_set:
                excluded_count += 1
                continue
              all_files.append((filename,mtime,num_rows))
          i += 1

        filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
        filtered_filenames = []
        for filename in filenames:
          if filename in exclude_set:
            excluded_count += 1
            continue
          filtered_filenames.append(filename)
        filenames = filtered_filenames

        files_with_unknown_num_rows.extend(filenames)
        filenames = [(filename,os.path.getmtime(filename)) for filename in filenames]
        all_files.extend(filenames)
  print("Total number of files: %d" % len(all_files), flush=True)
  print("Total number of files with unknown row count: %d" % len(files_with_unknown_num_rows), flush=True)
  print("Excluded count: %d" % excluded_count, flush=True)

  with TimeStuff("Sorting"):
    all_files.sort(key=(lambda x: x[1]), reverse=False)

  with TimeStuff("Computing rows for unsummarized files"):
    with multiprocessing.Pool(num_processes) as pool:
      results = pool.map(compute_num_rows,files_with_unknown_num_rows)
      results = dict(results)
      for i in range(len(all_files)):
        info = all_files[i]
        if len(info) < 3:
          all_files[i] = (info[0], info[1], results[info[0]])

  files_with_row_range = []
  num_rows_total = 0 #Number of data rows
  num_random_rows_capped = 0 #Number of random data rows, capped at min_rows - we never keep more than min_rows many data rows if they're from random.
  num_postrandom_rows = 0 #Number of NON-random rows

  #How far offset do we start on the power-law window tail? E.g. what number of postrandom rows do we need before the window size grows by a factor
  #of 2^(taper_window_exponent)? For now, we set it equal to the min rows
  if taper_window_scale is not None:
    window_taper_offset = taper_window_scale
  else:
    window_taper_offset = min_rows

  def num_usable_rows():
    global num_random_rows_capped
    global num_postrandom_rows
    return num_random_rows_capped + num_postrandom_rows
  def num_desired_rows():
    #Every post-random row moves one row beyond window_taper_offset
    power_law_x = num_usable_rows() - min_rows + window_taper_offset + add_to_window
    #Apply power law and correct for window_taper_offset so we're still anchored at 0
    unscaled_power_law = (power_law_x ** taper_window_exponent) - (window_taper_offset ** taper_window_exponent)
    #Scale so that we have an initial derivative of 1
    scaled_power_law = unscaled_power_law / (taper_window_exponent * (window_taper_offset ** (taper_window_exponent-1)))
    #Scale so that we have the desired initial slope, and add back the minimum random rows
    return int(scaled_power_law * expand_window_per_row + min_rows)

  for (filename,mtime,num_rows) in all_files:
    if num_rows <= 0:
      continue
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
  with TimeStuff("Computing desired rows"):
    for i in range(len(files_with_row_range)):
      (filename,(start_row,end_row)) = files_with_row_range[i]

      # This could happen if the .summary.json file is inaccurate after file deletions
      if not os.path.exists(filename):
        continue

      desired_input_files.append((filename,end_row-start_row))
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
  print("Writing %d output files with %d kept / %d desired rows" % (num_out_files, approx_rows_to_keep, desired_num_rows))

  def clean_tmp_dirs():
    for tmp_dir in out_tmp_dirs:
      if os.path.exists(tmp_dir):
        print("Cleaning up tmp dir: " + tmp_dir)
        shutil.rmtree(tmp_dir)

  clean_tmp_dirs()
  for tmp_dir in out_tmp_dirs:
    os.mkdir(tmp_dir)

  desired_input_file_groups = []
  group_size_so_far = 0
  group_so_far = []
  for (input_file,num_rows_in_file) in desired_input_files:
    if num_rows_in_file <= 0:
      continue
    group_so_far.append(input_file)
    group_size_so_far += num_rows_in_file
    if group_size_so_far >= worker_group_size:
      desired_input_file_groups.append(group_so_far)
      group_so_far = []
      group_size_so_far = 0
  if group_size_so_far > 0:
    desired_input_file_groups.append(group_so_far)
    group_so_far = []
    group_size_so_far = 0
  print("Grouping %d input files into %d sharding groups" % (len(desired_input_files),len(desired_input_file_groups)),flush=True)

  with multiprocessing.Pool(num_processes) as pool:
    with TimeStuff("Sharding"):
      shard_results = pool.starmap(shardify, [
        (input_idx, desired_input_file_groups[input_idx], num_out_files, out_tmp_dirs, keep_prob) for input_idx in range(len(desired_input_file_groups))
      ])

    with TimeStuff("Merging"):
      num_shards_to_merge = len(desired_input_file_groups)
      merge_results = pool.starmap(merge_shards, [
        (out_files[idx],num_shards_to_merge,out_tmp_dirs[idx],batch_size,ensure_batch_multiple) for idx in range(len(out_files))
      ])
    print("Mumber of rows by output file:",flush=True)
    print(list(zip(out_files,merge_results)),flush=True)
    sys.stdout.flush()

  clean_tmp_dirs()

  dump_value = {
    "range": (min(start_row for (filename,(start_row,end_row)) in desired_input_files_with_row_range),
              max(end_row for (filename,(start_row,end_row)) in desired_input_files_with_row_range))
  }

  with open(out_dir + ".json", 'w') as f:
    json.dump(dump_value, f)
