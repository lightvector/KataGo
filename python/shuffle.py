#!/usr/bin/python3
# Shuffles npz selfplay data for training, choosing a window size based on a power law.
# Run 'python shuffle.py --help' for details on how the window size is chosen and how to use this script.
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
import hashlib
import datetime
import gc

import multiprocessing

import numpy as np

def assert_keys(npz,include_meta):
    keys = [
        "binaryInputNCHWPacked",
        "globalInputNC",
        "policyTargetsNCMove",
        "globalTargetsNC",
        "scoreDistrN",
        "valueTargetsNCHW",
    ]
    if include_meta:
        keys.append("metadataInputNC")
    assert(set(npz.keys()) == set(keys))

def is_temp_npz_like(filename):
    return "_" in filename

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

def shardify(input_idx, input_file_group, num_out_files, out_tmp_dirs, keep_prob, include_meta):
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(4)])

    assert(len(input_file_group) > 0)
    num_files_not_found = 0

    if len(input_file_group) == 1:
        try:
            with np.load(input_file_group[0]) as npz:
                assert_keys(npz,include_meta)
                ###
                # WARNING - if adding anything here, also add it to joint_shuffle below!
                ###
                binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
                globalInputNC = npz["globalInputNC"]
                policyTargetsNCMove = npz["policyTargetsNCMove"]
                globalTargetsNC = npz["globalTargetsNC"]
                scoreDistrN = npz["scoreDistrN"]
                valueTargetsNCHW = npz["valueTargetsNCHW"]
                metadataInputNC = npz["metadataInputNC"] if include_meta else None
        except FileNotFoundError:
            num_files_not_found += 1
            print("WARNING: file not found by shardify: ", input_file_group[0])
            return num_files_not_found # Early quit since we don't know shapes
    else:
        binaryInputNCHWPackedList = []
        globalInputNCList = []
        policyTargetsNCMoveList = []
        globalTargetsNCList = []
        scoreDistrNList = []
        valueTargetsNCHWList = []
        metadataInputNCList = []

        for input_file in input_file_group:
            try:
                with np.load(input_file) as npz:
                    assert_keys(npz,include_meta)
                    binaryInputNCHWPackedList.append(npz["binaryInputNCHWPacked"])
                    globalInputNCList.append(npz["globalInputNC"])
                    policyTargetsNCMoveList.append(npz["policyTargetsNCMove"])
                    globalTargetsNCList.append(npz["globalTargetsNC"])
                    scoreDistrNList.append(npz["scoreDistrN"])
                    valueTargetsNCHWList.append(npz["valueTargetsNCHW"])
                    metadataInputNCList.append(npz["metadataInputNC"] if include_meta else None)
            except FileNotFoundError:
                num_files_not_found += 1
                print("WARNING: file not found by shardify: ", input_file)
                pass
        if len(binaryInputNCHWPackedList) <= 0:
            return num_files_not_found # Early quit since we don't know shapes

        binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackedList,axis=0)
        globalInputNC = np.concatenate(globalInputNCList,axis=0)
        policyTargetsNCMove = np.concatenate(policyTargetsNCMoveList,axis=0)
        globalTargetsNC = np.concatenate(globalTargetsNCList,axis=0)
        scoreDistrN = np.concatenate(scoreDistrNList,axis=0)
        valueTargetsNCHW = np.concatenate(valueTargetsNCHWList,axis=0)
        metadataInputNC = np.concatenate(metadataInputNCList,axis=0) if include_meta else None

    num_rows_to_keep = binaryInputNCHWPacked.shape[0]
    assert(globalInputNC.shape[0] == num_rows_to_keep)
    assert(policyTargetsNCMove.shape[0] == num_rows_to_keep)
    assert(globalTargetsNC.shape[0] == num_rows_to_keep)
    assert(scoreDistrN.shape[0] == num_rows_to_keep)
    assert(valueTargetsNCHW.shape[0] == num_rows_to_keep)
    assert(metadataInputNC.shape[0] == num_rows_to_keep if include_meta else True)

    if keep_prob < 1.0:
        num_rows_to_keep = min(num_rows_to_keep,int(round(num_rows_to_keep * keep_prob)))

    if include_meta:
        [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW,metadataInputNC] = (
            joint_shuffle_take_first_n(
                num_rows_to_keep,
                [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW,metadataInputNC]
            )
        )
    else:
        [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW] = (
            joint_shuffle_take_first_n(
                num_rows_to_keep,
                [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW]
            )
        )

    assert(binaryInputNCHWPacked.shape[0] == num_rows_to_keep)
    assert(globalInputNC.shape[0] == num_rows_to_keep)
    assert(policyTargetsNCMove.shape[0] == num_rows_to_keep)
    assert(globalTargetsNC.shape[0] == num_rows_to_keep)
    assert(scoreDistrN.shape[0] == num_rows_to_keep)
    assert(valueTargetsNCHW.shape[0] == num_rows_to_keep)
    assert(metadataInputNC.shape[0] == num_rows_to_keep if include_meta else True)

    rand_assts = np.random.randint(num_out_files,size=[num_rows_to_keep])
    counts = np.bincount(rand_assts,minlength=num_out_files)
    countsums = np.cumsum(counts)
    assert(countsums[len(countsums)-1] == num_rows_to_keep)

    # if input_idx % 29 == 0:
    #   print("%s: Shardify writing... (mem usage %dMB)" % (str(datetime.datetime.now()),memusage_mb()), flush=True)

    for out_idx in range(num_out_files):
        start = countsums[out_idx]-counts[out_idx]
        stop = countsums[out_idx]
        if include_meta:
            np.savez_compressed(
                os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
                binaryInputNCHWPacked = binaryInputNCHWPacked[start:stop],
                globalInputNC = globalInputNC[start:stop],
                policyTargetsNCMove = policyTargetsNCMove[start:stop],
                globalTargetsNC = globalTargetsNC[start:stop],
                scoreDistrN = scoreDistrN[start:stop],
                valueTargetsNCHW = valueTargetsNCHW[start:stop],
                metadataInputNC = metadataInputNC[start:stop],
            )
        else:
            np.savez_compressed(
                os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
                binaryInputNCHWPacked = binaryInputNCHWPacked[start:stop],
                globalInputNC = globalInputNC[start:stop],
                policyTargetsNCMove = policyTargetsNCMove[start:stop],
                globalTargetsNC = globalTargetsNC[start:stop],
                scoreDistrN = scoreDistrN[start:stop],
                valueTargetsNCHW = valueTargetsNCHW[start:stop],
            )
    return num_files_not_found

def merge_shards(filename, num_shards_to_merge, out_tmp_dir, batch_size, ensure_batch_multiple, output_npz, include_meta):
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(5)])

    if output_npz:
        record_writer = None
    else:
        assert False, "No longer supports outputting tensorflow data"

    binaryInputNCHWPackeds = []
    globalInputNCs = []
    policyTargetsNCMoves = []
    globalTargetsNCs = []
    scoreDistrNs = []
    valueTargetsNCHWs = []
    metadataInputNCs = []

    for input_idx in range(num_shards_to_merge):
        shard_filename = os.path.join(out_tmp_dir, str(input_idx) + ".npz")
        try:
            with np.load(shard_filename) as npz:
                assert_keys(npz,include_meta)

                binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
                globalInputNC = npz["globalInputNC"]
                policyTargetsNCMove = npz["policyTargetsNCMove"]
                globalTargetsNC = npz["globalTargetsNC"]
                scoreDistrN = npz["scoreDistrN"]
                valueTargetsNCHW = npz["valueTargetsNCHW"]
                metadataInputNC = npz["metadataInputNC"] if include_meta else None

                binaryInputNCHWPackeds.append(binaryInputNCHWPacked)
                globalInputNCs.append(globalInputNC)
                policyTargetsNCMoves.append(policyTargetsNCMove)
                globalTargetsNCs.append(globalTargetsNC)
                scoreDistrNs.append(scoreDistrN)
                valueTargetsNCHWs.append(valueTargetsNCHW)
                metadataInputNCs.append(metadataInputNC)
        except FileNotFoundError:
            print("WARNING: Empty shard in merge_shards for shard :", input_idx, filename)

    if len(binaryInputNCHWPackeds) <= 0:
        print("WARNING: empty merge file: ", filename)
        return 0

    ###
    # WARNING - if adding anything here, also add it to joint_shuffle below!
    ###
    binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackeds)
    globalInputNC = np.concatenate(globalInputNCs)
    policyTargetsNCMove = np.concatenate(policyTargetsNCMoves)
    globalTargetsNC = np.concatenate(globalTargetsNCs)
    scoreDistrN = np.concatenate(scoreDistrNs)
    valueTargetsNCHW = np.concatenate(valueTargetsNCHWs)
    metadataInputNC = np.concatenate(metadataInputNCs) if include_meta else None

    num_rows = binaryInputNCHWPacked.shape[0]
    assert(globalInputNC.shape[0] == num_rows)
    assert(policyTargetsNCMove.shape[0] == num_rows)
    assert(globalTargetsNC.shape[0] == num_rows)
    assert(scoreDistrN.shape[0] == num_rows)
    assert(valueTargetsNCHW.shape[0] == num_rows)
    assert(metadataInputNC.shape[0] == num_rows if include_meta else True)

    if include_meta:
        [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW,metadataInputNC] = (
            joint_shuffle_take_first_n(
                num_rows,
                [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW,metadataInputNC],
            )
        )
    else:
        [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW] = (
            joint_shuffle_take_first_n(
                num_rows,
                [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW],
            )
        )

    assert(binaryInputNCHWPacked.shape[0] == num_rows)
    assert(globalInputNC.shape[0] == num_rows)
    assert(policyTargetsNCMove.shape[0] == num_rows)
    assert(globalTargetsNC.shape[0] == num_rows)
    assert(scoreDistrN.shape[0] == num_rows)
    assert(valueTargetsNCHW.shape[0] == num_rows)
    assert(metadataInputNC.shape[0] == num_rows if include_meta else True)

    # print("%s: Merge writing... (mem usage %dMB)" % (str(datetime.datetime.now()),memusage_mb()), flush=True)

    # Just truncate and lose the batch at the end, it's fine
    num_batches = (num_rows // (batch_size * ensure_batch_multiple)) * ensure_batch_multiple
    if output_npz:
        start = 0
        stop = num_batches*batch_size
        if include_meta:
            np.savez_compressed(
                filename,
                binaryInputNCHWPacked = binaryInputNCHWPacked[start:stop],
                globalInputNC = globalInputNC[start:stop],
                policyTargetsNCMove = policyTargetsNCMove[start:stop],
                globalTargetsNC = globalTargetsNC[start:stop],
                scoreDistrN = scoreDistrN[start:stop],
                valueTargetsNCHW = valueTargetsNCHW[start:stop],
                metadataInputNC = metadataInputNC[start:stop],
            )
        else:
            np.savez_compressed(
                filename,
                binaryInputNCHWPacked = binaryInputNCHWPacked[start:stop],
                globalInputNC = globalInputNC[start:stop],
                policyTargetsNCMove = policyTargetsNCMove[start:stop],
                globalTargetsNC = globalTargetsNC[start:stop],
                scoreDistrN = scoreDistrN[start:stop],
                valueTargetsNCHW = valueTargetsNCHW[start:stop],
            )
    else:
        assert False, "No longer supports outputting tensorflow data"

    jsonfilename = os.path.splitext(filename)[0] + ".json"
    with open(jsonfilename,"w") as f:
        json.dump({"num_rows":num_rows,"num_batches":num_batches},f)

    if record_writer is not None:
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
    try:
        npheaders = get_numpy_npz_headers(filename)
    except PermissionError:
        print("WARNING: No permissions for reading file: ", filename)
        return (filename,None)
    except zipfile.BadZipFile:
        print("WARNING: Bad zip file: ", filename)
        return (filename,None)
    if npheaders is None or len(npheaders) <= 0:
        print("WARNING: bad npz headers for file: ", filename)
        return (filename,None)

    if "binaryInputNCHWPacked" in npheaders:
        (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
    else:
        (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked.npy"]
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
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False,formatter_class=argparse.RawTextHelpFormatter,description="""
    Shuffle data files!

    This shuffle script is designed for ongoing self-play training. It shuffles the most recent window of data among the data it's provided. It chooses the window size dynamically based on the total amount of data in the run so far, assuming that the directories provided contain all of the data for the run so far. If you don't actually have all of the data, e.g. you've archived or deleted the older data, or else want to compute the window size as if there were more data, use -add-to-data-rows.

    The window size is a power law based on the number of rows in the run N:
      WINDOWSIZE(N) = (N^EXPONENT - MIN_ROWS^EXPONENT) / (EXPONENT * MIN_ROWS^(EXPONENT-1)) * INITIAL_WINDOW_PER_ROW + MIN_ROWS

    given arguments:
      -taper-window-exponent EXPONENT \\
      -expand-window-per-row INITIAL_WINDOW_PER_ROW \\
      -min-rows MIN_ROWS  (default 250k)

    This may look a bit complex, but basically it is simply the power law N^EXPONENT with shifting and scaling such that:
    WINDOWSIZE(MIN_ROWS) = MIN_ROWS
    (dWINDOWSIZE/dN)(MIN_ROWS) = INITIAL_WINDOW_PER_ROW

    Reasonable arguments similar to those used for KataGo's main runs would be
      -taper-window-exponent 0.65 or 0.675 \\
      -expand-window-per-row 0.4 \\
      -min-rows 250000 (default)

    If you want to control the "scale" of the power law differently than the min rows, you can specify -taper-window-scale as well.
    There is also a bit of a hack to cap the number of random rows (rows generated by random play without a neural net), since random row generation at the start of a run can be very fast due to not hitting the GPU, and overpopulate the run.

    Additionally, NOT all of the shuffled window is output, only a random shuffled 20M rows will be kept. Adjust this using -keep-target-rows. The intention is that this script will be repeatedly run as new data comes in, such that well before train.py would need more than 20M rows, the data would have been shuffled again and a new random 20M rows chosen.

    If you are NOT doing ongoing self-play training, but simply want to shuffle an entire dataset (not just a window of it) and want to output all of it (not just 20M of it) then you can use arguments like:
      -taper-window-exponent 1.0 \\
      -expand-window-per-row 1.0 \\
      -keep-target-rows SOME_VERY_LARGE_NUMBER

    If you ARE doing ongoing self-play training, but want a fixed window size, then you can use arguments like:
      -min-rows YOUR_DESIRED_SIZE \\
      -taper-window-exponent 1.0 \\
      -expand-window-per-row 0.0
    """)
    parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')

    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    optional_args.add_argument('-min-rows', type=int, required=False, help='Minimum training rows to use, default 250k')
    optional_args.add_argument('-max-rows', type=int, required=False, help='Maximum training rows to use, default unbounded')
    optional_args.add_argument('-keep-target-rows', type=int, required=False, help='Target number of rows to actually keep in the final data set, default 20M')
    required_args.add_argument('-expand-window-per-row', type=float, required=True, help='Beyond min rows, initially expand the window by this much every post-random data row')
    required_args.add_argument('-taper-window-exponent', type=float, required=True, help='Make the window size asymtotically grow as this power of the data rows')
    optional_args.add_argument('-taper-window-scale', type=float, required=False, help='The scale at which the power law applies, defaults to -min-rows')
    optional_args.add_argument('-add-to-data-rows', type=float, required=False, help='Compute the window size as if the number of data rows were this much larger/smaller')
    optional_args.add_argument('-add-to-window-size', type=float, required=False, help='DEPRECATED due to being misnamed name, use -add-to-data-rows')
    optional_args.add_argument('-summary-file', required=False, help='Summary json file for directory contents')
    required_args.add_argument('-out-dir', required=True, help='Dir to output training files')
    required_args.add_argument('-out-tmp-dir', required=True, help='Dir to use as scratch space')
    optional_args.add_argument('-approx-rows-per-out-file', type=int, required=False, default=70000, help='Number of rows per output file, default 70k')
    required_args.add_argument('-num-processes', type=int, required=True, help='Number of multiprocessing processes for shuffling in parallel')
    required_args.add_argument('-batch-size', type=int, required=True, help='Batch size to write training examples in')
    optional_args.add_argument('-ensure-batch-multiple', type=int, required=False, help='Ensure each file is a multiple of this many batches')
    optional_args.add_argument('-worker-group-size', type=int, required=False, default=80000, help='Internally, target having many rows per parallel sharding worker (doesnt affect merge)')
    optional_args.add_argument('-exclude', required=False, help='Text file with npzs to ignore, one per line')
    optional_args.add_argument('-exclude-prefix', required=False, help='Prefix to concat to lines in exclude to produce the full file path')
    optional_args.add_argument('-exclude-basename', required=False, action="store_true", help='Consider an exclude to match if basename matches')
    optional_args.add_argument('-only-include-md5-path-prop-lbound', type=float, required=False, help='Just before sharding, include only filepaths hashing to float >= this')
    optional_args.add_argument('-only-include-md5-path-prop-ubound', type=float, required=False, help='Just before sharding, include only filepaths hashing to float < this')
    optional_args.add_argument('-output-npz', action="store_true", required=False, help='Output results as npz files')
    optional_args.add_argument('-include-meta', action="store_true", required=False, help='Include sgf metadata inputs')

    args = parser.parse_args()
    dirs = args.dirs
    min_rows = args.min_rows
    max_rows = args.max_rows
    keep_target_rows = args.keep_target_rows
    expand_window_per_row = args.expand_window_per_row
    taper_window_exponent = args.taper_window_exponent
    taper_window_scale = args.taper_window_scale
    add_to_data_rows = args.add_to_data_rows
    if args.add_to_data_rows is not None and args.add_to_window_size is not None:
        print("Cannot specify both -add-to-data-rows and -add-to-window-size. Please use only -add-to-data-rows, -add-to-window-size is deprecated")
    if args.add_to_data_rows is None and args.add_to_window_size is not None:
        print("WARNING: -add-to-window-size is deprecated due to being misnamed, use -add-to-data-rows")
        add_to_data_rows = args.add_to_window_size

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
    exclude = args.exclude
    exclude_prefix = args.exclude_prefix
    if exclude_prefix is None:
        exclude_prefix = ""
    exclude_basename = args.exclude_basename
    only_include_md5_path_prop_lbound = args.only_include_md5_path_prop_lbound
    only_include_md5_path_prop_ubound = args.only_include_md5_path_prop_ubound
    output_npz = args.output_npz
    include_meta = args.include_meta

    if min_rows is None:
        print("NOTE: -min-rows was not specified, defaulting to requiring 250K rows before shuffling.")
        min_rows = 250000
    if keep_target_rows is None:
        print("NOTE: -keep-target-rows was not specified, defaulting to sampling a random 20M rows out of the computed window.")
        print("If you intended to shuffle the whole dataset instead, pass in -keep-target-rows <very large number>")
        keep_target_rows = 20000000
    if add_to_data_rows is None:
        add_to_data_rows = 0

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
                raise RuntimeError("Could not load exclude file")

    # If excluding basenames, also add them to the set
    if exclude_basename:
        basenames = [os.path.basename(path) for path in exclude_set]
        exclude_set.update(basenames)

    all_files = []
    files_with_unknown_num_rows = []
    excluded_count = 0
    excluded_due_to_excludes_count = 0
    tempfilelike_count = 0
    with TimeStuff("Finding files"):
        for d in dirs:
            for (path,dirnames,filenames) in os.walk(d, followlinks=True):
                i = 0
                while i < len(dirnames):
                    dirname = dirnames[i]
                    summary_data = summary_data_by_dirpath.get(os.path.abspath(os.path.join(path, dirname)))
                    if summary_data is not None:
                        filename_mtime_num_rowss = summary_data["filename_mtime_num_rowss"]
                        del dirnames[i]
                        i -= 1
                        for (filename,mtime,num_rows) in filename_mtime_num_rowss:
                            if is_temp_npz_like(filename):
                                # print("WARNING: file looks like a temp file, treating as exclude: ", os.path.join(path,dirname,filename))
                                excluded_count += 1
                                tempfilelike_count += 1
                                continue
                            if exclude_basename and os.path.basename(filename) in exclude_set:
                                excluded_count += 1
                                excluded_due_to_excludes_count += 1
                                continue
                            filename = os.path.join(path,dirname,filename)
                            if not exclude_basename and filename in exclude_set:
                                excluded_count += 1
                                excluded_due_to_excludes_count += 1
                                continue
                            if num_rows is None:
                                print("WARNING: Skipping bad rowless file, treating as exclude: ", filename)
                                excluded_count += 1
                                continue
                            all_files.append((filename,mtime,num_rows))
                    i += 1

                filtered_filenames = []
                for filename in filenames:
                    if not filename.endswith(".npz"):
                        continue
                    if is_temp_npz_like(filename):
                        # print("WARNING: file looks like a temp file, treating as exclude: ", os.path.join(path,filename))
                        excluded_count += 1
                        tempfilelike_count += 1
                        continue
                    if exclude_basename and os.path.basename(filename) in exclude_set:
                        excluded_count += 1
                        excluded_due_to_excludes_count += 1
                        continue
                    filename = os.path.join(path,filename)
                    if not exclude_basename and filename in exclude_set:
                        excluded_count += 1
                        excluded_due_to_excludes_count += 1
                        continue
                    filtered_filenames.append(filename)
                filenames = filtered_filenames

                files_with_unknown_num_rows.extend(filenames)
                filenames = [(filename,os.path.getmtime(filename)) for filename in filenames]
                all_files.extend(filenames)
    print("Total number of files: %d" % len(all_files), flush=True)
    print("Total number of files with unknown row count: %d" % len(files_with_unknown_num_rows), flush=True)
    print("Excluded count: %d" % excluded_count, flush=True)
    print("Excluded count due to looking like temp file: %d" % tempfilelike_count, flush=True)
    print("Excluded count due to cmdline excludes file: %d" % excluded_due_to_excludes_count, flush=True)

    print("GC collect", flush=True)
    del summary_data_by_dirpath
    gc.collect()

    with TimeStuff("Sorting"):
        all_files.sort(key=(lambda x: x[1]), reverse=False)

    # Wait a few seconds just in case to limit the chance of filesystem races, now that we know exactly
    # the set of filenames we want
    time.sleep(3)

    with TimeStuff("Computing rows for unsummarized files"):
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(compute_num_rows,files_with_unknown_num_rows)
            results = dict(results)
            for i in range(len(all_files)):
                info = all_files[i]
                if len(info) < 3:
                    num_rows = results[info[0]]
                    all_files[i] = (info[0], info[1], num_rows)

    num_rows_total = 0 # Number of data rows
    num_random_rows_capped = 0 # Number of random data rows, capped at min_rows - we never keep more than min_rows many data rows if they're from random.
    num_postrandom_rows = 0 # Number of NON-random rows

    # How far offset do we start on the power-law window tail? E.g. what number of postrandom rows do we need before the window size grows by a factor
    # of 2^(taper_window_exponent)? For now, we set it equal to the min rows
    if taper_window_scale is not None:
        window_taper_offset = taper_window_scale
    else:
        window_taper_offset = min_rows

    def num_usable_rows():
        global num_random_rows_capped
        global num_postrandom_rows
        return num_random_rows_capped + num_postrandom_rows
    def num_desired_rows():
        # Every post-random row moves one row beyond window_taper_offset
        power_law_x = num_usable_rows() - min_rows + window_taper_offset + add_to_data_rows
        # Apply power law and correct for window_taper_offset so we're still anchored at 0
        unscaled_power_law = (power_law_x ** taper_window_exponent) - (window_taper_offset ** taper_window_exponent)
        # Scale so that we have an initial derivative of 1
        scaled_power_law = unscaled_power_law / (taper_window_exponent * (window_taper_offset ** (taper_window_exponent-1)))
        # Scale so that we have the desired initial slope, and add back the minimum random rows
        return int(scaled_power_law * expand_window_per_row + min_rows)

    with TimeStuff("Processing found files"):
        for (filename,mtime,num_rows) in all_files:
            if num_rows is None:
                print("WARNING: Skipping bad file: ", filename)
                continue
            if num_rows <= 0:
                continue
            num_rows_total += num_rows
            if "random/tdata/" not in filename and "random\\tdata\\" not in filename:
                num_postrandom_rows += num_rows
            else:
                num_random_rows_capped = min(num_random_rows_capped + num_rows, min_rows)

    if os.path.exists(out_dir):
        raise Exception(out_dir + " already exists")
    os.mkdir(out_dir)

    if num_rows_total <= 0:
        print("No rows found")
        sys.exit(0)

    # If we don't have enough rows, then quit out
    if num_rows_total < min_rows:
        print("Not enough rows, only %d (fewer than %d)" % (num_rows_total,min_rows))
        sys.exit(0)

    print("Total rows found: %d (%d usable)" % (num_rows_total,num_usable_rows()), flush=True)

    # Reverse so that recent files are first
    all_files.reverse()

    # Now assemble only the files we need to hit our desired window size
    desired_num_rows = num_desired_rows()
    desired_num_rows = max(desired_num_rows,min_rows)
    desired_num_rows = min(desired_num_rows,max_rows) if max_rows is not None else desired_num_rows
    print("Desired num rows: %d / %d" % (desired_num_rows,num_rows_total), flush=True)

    desired_input_files = []
    min_start_row = num_rows_total
    max_end_row = num_rows_total
    num_rows_used = 0
    print_stride = 1 + len(all_files) // 80
    end_row = num_rows_total
    with TimeStuff("Computing desired rows"):
        for i in range(len(all_files)):
            (filename,mtime,num_rows) = all_files[i]

            # This could happen if the .summary.json file is inaccurate after file deletions
            # Actually we just handle that in shardify - and accept that it might make our window slightly not far back enough
            # if not os.path.exists(filename):
            #   continue

            if num_rows is not None and num_rows > 0:
                desired_input_files.append((filename,num_rows))
                start_row = end_row - num_rows
                min_start_row = min(start_row, min_start_row)
                num_rows_used += num_rows
                end_row -= num_rows
            else:
                start_row = end_row

            if i % print_stride == 0 or num_rows_used >= desired_num_rows:
                print("Using: %s (%d-%d) (%d/%d desired rows)" % (filename,start_row,end_row,num_rows_used,desired_num_rows), flush=True)
            if num_rows_used >= desired_num_rows:
                break

    print("Finally, using: (%d-%d) (%d/%d desired rows)" % (min_start_row,max_end_row,num_rows_used,desired_num_rows), flush=True)

    print("GC collect", flush=True)
    del all_files
    gc.collect()

    np.random.seed()
    np.random.shuffle(desired_input_files)

    approx_rows_to_keep = num_rows_used
    if keep_target_rows is not None:
        approx_rows_to_keep = min(approx_rows_to_keep, keep_target_rows)
    keep_prob = approx_rows_to_keep / num_rows_used

    num_out_files = int(round(approx_rows_to_keep / approx_rows_per_out_file))
    num_out_files = max(num_out_files,1)

    if output_npz:
        out_files = [os.path.join(out_dir, "data%d.npz" % i) for i in range(num_out_files)]
    else:
        assert False, "No longer supports outputting tensorflow data"

    out_tmp_dirs = [os.path.join(out_tmp_dir, "tmp.shuf%d" % i) for i in range(num_out_files)]
    print("Writing %d output files with %d kept / %d desired rows" % (num_out_files, approx_rows_to_keep, desired_num_rows), flush=True)

    def clean_tmp_dirs():
        for tmp_dir in out_tmp_dirs:
            if os.path.exists(tmp_dir):
                print("Cleaning up tmp dir: " + tmp_dir)
                shutil.rmtree(tmp_dir)

    clean_tmp_dirs()
    for tmp_dir in out_tmp_dirs:
        os.makedirs(tmp_dir,exist_ok=True)

    num_rows_in_desired_files = 0
    if only_include_md5_path_prop_lbound is not None or only_include_md5_path_prop_ubound is not None:
        new_desired_input_files = []
        for (input_file,num_rows_in_file) in desired_input_files:
            input_file_base = os.path.basename(input_file)
            hashfloat = int("0x"+hashlib.md5(str(input_file_base).encode('utf-8')).hexdigest()[:13],16) / 2 ** 52
            ok = True
            if only_include_md5_path_prop_lbound is not None and hashfloat < only_include_md5_path_prop_lbound:
                ok = False
            if only_include_md5_path_prop_ubound is not None and hashfloat >= only_include_md5_path_prop_ubound:
                ok = False
            if ok:
                new_desired_input_files.append((input_file,num_rows_in_file))
                num_rows_in_desired_files += num_rows_in_file
        print("Due to only_include_md5, filtering down to %d/%d files" % (len(new_desired_input_files),len(desired_input_files)))
        desired_input_files = new_desired_input_files
    else:
        for (input_file,num_rows_in_file) in desired_input_files:
            num_rows_in_desired_files += num_rows_in_file

    if len(desired_input_files) <= 0:
        print("No files after filtering for desired range")
        sys.exit(0)
    if num_rows_in_desired_files <= 0:
        print("No rows in desired files")
        sys.exit(0)

    # Clump files into sharding groups. More efficient if shuffling a ton of small npz files
    # since we aren't doing separate tasks for every individual file but rather handling a bunch
    # of files at once, and also makes chunkier shards on disk when it comes time to shuffle.
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
                (input_idx, desired_input_file_groups[input_idx], num_out_files, out_tmp_dirs, keep_prob, include_meta)
                for input_idx in range(len(desired_input_file_groups))
            ])

        with TimeStuff("Merging"):
            num_shards_to_merge = len(desired_input_file_groups)
            merge_results = pool.starmap(merge_shards, [
                (out_files[idx],num_shards_to_merge,out_tmp_dirs[idx],batch_size,ensure_batch_multiple,output_npz,include_meta)
                for idx in range(len(out_files))
            ])
        print("Number of rows by output file:",flush=True)
        print(list(zip(out_files,merge_results)),flush=True)
        sys.stdout.flush()

    clean_tmp_dirs()

    dump_value = {
        "range": (min_start_row, max_end_row)
    }

    with open(out_dir + ".json", 'w') as f:
        json.dump(dump_value, f)
