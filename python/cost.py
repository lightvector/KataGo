#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import math
import time
import logging
import zipfile

import multiprocessing

import numpy as np

keys = [
  "binaryInputNCHWPacked",
  "globalInputNC",
  "policyTargetsNCMove",
  "globalTargetsNC",
  "scoreDistrN",
  "selfBonusScoreN",
  "valueTargetsNCHW"
]

def compute_stats(input_file):
  npz = np.load(input_file)
  assert(set(npz.keys()) == set(keys))

  #binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
  #globalInputNC = npz["globalInputNC"]
  policyTargetsNCMove = npz["policyTargetsNCMove"]
  globalTargetsNC = npz["globalTargetsNC"]
  #scoreDistrN = npz["scoreDistrN"]
  #selfBonusScoreN = npz["selfBonusScoreN"]
  #valueTargetsNCHW = npz["valueTargetsNCHW"]

  num_rows = policyTargetsNCMove.shape[0]
  #assert(binaryInputNCHWPacked.shape[0] == num_rows)
  #assert(globalInputNC.shape[0] == num_rows)
  assert(policyTargetsNCMove.shape[0] == num_rows)
  assert(globalTargetsNC.shape[0] == num_rows)
  #assert(scoreDistrN.shape[0] == num_rows)
  #assert(selfBonusScoreN.shape[0] == num_rows)
  #assert(valueTargetsNCHW.shape[0] == num_rows)

  #Total number of visits - NOTE overestimate due to post-reduction of noise moves
  num_visits = np.sum(policyTargetsNCMove[:,0,:])
  #Number of reusable visits - NOTE overestimate since we won't always choose this move as best
  #and on the next turn we rebuild the tree from nn cache but might not always hit the same branches.
  is_reusable = 1.0 - globalTargetsNC[:,53] #avoid counting side position searches for reusability
  num_reusable_visits = np.sum(np.max(policyTargetsNCMove[:,0,:],axis=1) * is_reusable)

  if "random" in input_file:
    architecture = "random"
    per_eval_cost = 0
  elif "b6c96" in input_file:
    architecture = "b6c96"
    per_eval_cost = 6*96*96
  elif "b10c128" in input_file:
    architecture = "b10c128"
    per_eval_cost = 10*128*128
  elif "b15c192" in input_file:
    architecture = "b15c192"
    per_eval_cost = 15*192*192
  elif "b20c256" in input_file:
    architecture = "b20c256"
    per_eval_cost = 20*256*256
  else:
    raise Exception("Could not determine neural net size from file: " + str(input_file))

  #In units of millions of 20x256 evals:
  cost = num_visits * per_eval_cost / (20*256*256 * 1000000)
  reusable_cost = num_reusable_visits * per_eval_cost / (20*256*256 * 1000000)

  return (architecture, num_rows, num_visits, num_reusable_visits, cost, reusable_cost)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Shuffle data files')
  parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')
  parser.add_argument('-num-processes', type=int, required=True, help='Number of multiprocessing processes')

  args = parser.parse_args()
  dirs = args.dirs
  num_processes = args.num_processes

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
  num_rows_total = 0 #Number of data rows

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

  if num_rows_total <= 0:
    print("No rows found")
    sys.exit(0)

  print("Total rows found: %d" % (num_rows_total))

  with multiprocessing.Pool(num_processes) as pool:
    print("Beginning calculation",flush=True)
    t0 = time.time()
    results = pool.starmap(compute_stats, [
      (filename,) for (filename,row_range) in files_with_row_range
    ])
    t1 = time.time()
    print("Done calculating:",flush=True)
    for ((filename,(start,stop)),result) in zip(files_with_row_range,results):
      (architecture, num_rows, num_visits, num_reusable_visits, cost, reusable_cost) = result
      print("%s,%s,%d,%d,%d,%f,%f,%f,%f" % (filename,architecture,start,stop,num_rows,num_visits,num_reusable_visits,cost,reusable_cost),flush=True)
    print("Time taken: " + str(t1-t0),flush=True)
    sys.stdout.flush()

