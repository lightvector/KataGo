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
from typing import Sequence

# Needs to be kept in sync with QVALUE_SPATIAL_TARGET_NUM_CHANNELS in trainingwrite.cpp C++ code among other places.
EXPECTED_Q_VALUE_TARGETS_NCMOVE_CHANNELS = 3

# Empirically measured per-row sizes for 19x19 data, used only for the rough resource-cost
# estimates printed by -dry-run-print-resource-cost (and informationally on normal runs).

# binaryInputNCHWPacked 1012, globalInputNC 76, policyTargetsNCMove 1448,
# globalTargetsNC 256, scoreDistrN 842, valueTargetsNCHW 1805  (sum = 5439, "required")
# + qValueTargetsNCMove 2172, + metadataInputNC 768 when those are included.
UNCOMPRESSED_BYTES_PER_ROW_REQUIRED_19 = 5439
UNCOMPRESSED_BYTES_PER_ROW_QVALUES_19 = 2172
UNCOMPRESSED_BYTES_PER_ROW_META_19 = 768
# Whole-file (large npz) compressed size as a fraction of uncompressed, measured on
# python/testdata/benchmark_data_1024.npz: ~0.118 with qvalues, ~0.109 without.
# Rounded up slightly up so disk estimate errs high rather than low.
COMPRESSED_FRACTION_LARGE_FILE = 0.12
# Small-shard compression "cliff": compressing tiny npz files is less efficient as rows drop.
# Measured benchmark_data_1024.npz fits well to:
#   compressed_bytes_per_row ~= asymptote + CLIFF_OVERHEAD_BYTES_PER_FILE / n_rows
# with asymptote = COMPRESSED_FRACTION_LARGE_FILE * uncompressed_bytes_per_row.
CLIFF_OVERHEAD_BYTES_PER_FILE = 1650

def assert_keys(npz, include_meta: bool, include_qvalues: bool):
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
    # We don't require qValueTargetsNCMove even if include_qvalues is True
    # since we'll handle missing values by filling with zeros
    expected_keys = set(keys)
    actual_keys = set(npz.keys())
    if "qValueTargetsNCMove" in actual_keys:
        expected_keys.add("qValueTargetsNCMove")
    assert actual_keys == expected_keys

def is_temp_npz_like(filename: str) -> bool:
    return "_" in filename

def assert_batch_dim_matches(arrs: Sequence[np.ndarray | None], assert_size_equals: int | None = None):
    # arrs[0] (binaryInputNCHWPacked) is always present and is used as the reference length.
    assert arrs[0] is not None
    for arr in arrs:
        if arr is not None:
            assert arr.shape[0] == arrs[0].shape[0]
    if assert_size_equals is not None:
        assert arrs[0].shape[0] == assert_size_equals

def assert_list_lengths_match(arrlists: Sequence[list[np.ndarray] | None]):
    # arrlists[0] (binaryInputNCHWPackedList) is always present and is the reference.
    assert arrlists[0] is not None
    for arrlist in arrlists:
        if arrlist is not None:
            assert len(arrlist) == len(arrlists[0])

def joint_shuffle_take_first_n(n: int, arrs: Sequence[np.ndarray | None]) -> list[np.ndarray | None]:
    assert_batch_dim_matches(arrs)
    perm = np.random.permutation(len(arrs[0]))
    perm = perm[:n]
    shuffled_arrs: list[np.ndarray | None] = []
    for arr in arrs:
        if arr is not None:
            shuffled_arrs.append(arr[perm])
        else:
            shuffled_arrs.append(None)
    return shuffled_arrs

def joint_concatenate(arrlists: Sequence[list[np.ndarray] | None]) -> list[np.ndarray | None]:
    assert_list_lengths_match(arrlists)
    if len(arrlists[0]) == 1:
        return [(arrlist[0] if arrlist is not None else None) for arrlist in arrlists]
    return [(np.concatenate(arrlist, axis=0) if arrlist is not None else None) for arrlist in arrlists]

def memusage_mb():
    return psutil.Process(os.getpid()).memory_info().rss // 1048576


def save_output_npz(
    filename: str,
    arrs: Sequence[np.ndarray | None],
    include_meta: bool,
    include_qvalues: bool,
    start: int,
    stop: int,
):
    assert len(arrs) == 8
    [binaryInputNCHWPacked,globalInputNC,policyTargetsNCMove,globalTargetsNC,scoreDistrN,valueTargetsNCHW,metadataInputNC,qValueTargetsNCMove] = (
        arrs
    )
    assert binaryInputNCHWPacked is not None
    assert globalInputNC is not None
    assert policyTargetsNCMove is not None
    assert globalTargetsNC is not None
    assert scoreDistrN is not None
    assert valueTargetsNCHW is not None
    assert (metadataInputNC is not None) == include_meta
    assert (qValueTargetsNCMove is not None) == include_qvalues

    save_dict = {
        "binaryInputNCHWPacked": binaryInputNCHWPacked[start:stop],
        "globalInputNC": globalInputNC[start:stop],
        "policyTargetsNCMove": policyTargetsNCMove[start:stop],
        "globalTargetsNC": globalTargetsNC[start:stop],
        "scoreDistrN": scoreDistrN[start:stop],
        "valueTargetsNCHW": valueTargetsNCHW[start:stop],
    }
    if metadataInputNC is not None:
        save_dict["metadataInputNC"] = metadataInputNC[start:stop]
    if qValueTargetsNCMove is not None:
        save_dict["qValueTargetsNCMove"] = qValueTargetsNCMove[start:stop]

    np.savez_compressed(
        filename,
        **save_dict
    )

def load_and_accumulate_input_contents(
    input_file: str,
    binaryInputNCHWPackedList: list[np.ndarray],
    globalInputNCList: list[np.ndarray],
    policyTargetsNCMoveList: list[np.ndarray],
    globalTargetsNCList: list[np.ndarray],
    scoreDistrNList: list[np.ndarray],
    valueTargetsNCHWList: list[np.ndarray],
    metadataInputNCList: list[np.ndarray] | None,
    qValueTargetsNCMoveList: list[np.ndarray] | None,
    include_meta: bool,
    include_qvalues: bool,
    fill_in_qvalues: bool
):
    with np.load(input_file) as npz:
        assert_keys(npz, include_meta, include_qvalues)
        binaryInputNCHWPackedList.append(npz["binaryInputNCHWPacked"])
        globalInputNCList.append(npz["globalInputNC"])
        policyTargetsNCMoveList.append(npz["policyTargetsNCMove"])
        globalTargetsNCList.append(npz["globalTargetsNC"])
        scoreDistrNList.append(npz["scoreDistrN"])
        valueTargetsNCHWList.append(npz["valueTargetsNCHW"])
        if metadataInputNCList is not None:
            metadataInputNCList.append(npz["metadataInputNC"])
        if qValueTargetsNCMoveList is not None:
            if fill_in_qvalues:
                if "qValueTargetsNCMove" in npz:
                    assert npz["qValueTargetsNCMove"].shape[1] == EXPECTED_Q_VALUE_TARGETS_NCMOVE_CHANNELS
                    qValueTargetsNCMoveList.append(npz["qValueTargetsNCMove"])
                else:
                    # Create zeros array with shape matching policyTargetsNCMove but with different C dimension
                    shape = list(npz["policyTargetsNCMove"].shape)
                    shape[1] = EXPECTED_Q_VALUE_TARGETS_NCMOVE_CHANNELS
                    qValueTargetsNCMoveList.append(np.zeros(shape, dtype=np.int16))
            else:
                qValueTargetsNCMoveList.append(npz["qValueTargetsNCMove"])


def shardify(
    input_idx: int,
    input_file_group: list[str],
    num_out_files: int,
    out_tmp_dirs: list[str],
    keep_prob: float,
    include_meta: bool,
    include_qvalues: bool,
    fill_in_qvalues: bool = True,
):
    # fill_in_qvalues=True is for reading raw input npzs that may lack qValueTargetsNCMove
    # (synthesize zeros). When re-sharding already-processed shard files (e.g. wave shards
    # produced by an earlier shardify pass), pass fill_in_qvalues=False since those files
    # already contain qValueTargetsNCMove.

    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(4)])

    assert len(input_file_group) > 0
    num_files_not_found = 0

    binaryInputNCHWPackedList = []
    globalInputNCList = []
    policyTargetsNCMoveList = []
    globalTargetsNCList = []
    scoreDistrNList = []
    valueTargetsNCHWList = []
    metadataInputNCList = [] if include_meta else None
    qValueTargetsNCMoveList = [] if include_qvalues else None

    for input_file in input_file_group:
        try:
            load_and_accumulate_input_contents(input_file,binaryInputNCHWPackedList,globalInputNCList,policyTargetsNCMoveList,globalTargetsNCList,scoreDistrNList,valueTargetsNCHWList,metadataInputNCList,qValueTargetsNCMoveList,include_meta,include_qvalues,fill_in_qvalues=fill_in_qvalues)

        except FileNotFoundError:
            num_files_not_found += 1
            print("WARNING: file not found by shardify: ", input_file)
            pass

    if len(binaryInputNCHWPackedList) <= 0:
        return num_files_not_found # Early quit since we don't know shapes

    concatenated_arrs = (
        joint_concatenate(
            [binaryInputNCHWPackedList,globalInputNCList,policyTargetsNCMoveList,globalTargetsNCList,scoreDistrNList,valueTargetsNCHWList,metadataInputNCList,qValueTargetsNCMoveList]
        )
    )

    assert_batch_dim_matches(concatenated_arrs)

    num_rows_to_keep = concatenated_arrs[0].shape[0]
    if keep_prob < 1.0:
        num_rows_to_keep = min(num_rows_to_keep,int(round(num_rows_to_keep * keep_prob)))

    concatenated_arrs = joint_shuffle_take_first_n(num_rows_to_keep, concatenated_arrs)
    assert_batch_dim_matches(concatenated_arrs, num_rows_to_keep)

    rand_assts = np.random.randint(num_out_files,size=[num_rows_to_keep])
    counts = np.bincount(rand_assts,minlength=num_out_files)
    countsums = np.cumsum(counts)
    assert(countsums[len(countsums)-1] == num_rows_to_keep)

    # if input_idx % 29 == 0:
    #   print("%s: Shardify writing... (mem usage %dMB)" % (str(datetime.datetime.now()),memusage_mb()), flush=True)

    for out_idx in range(num_out_files):
        start = countsums[out_idx]-counts[out_idx]
        stop = countsums[out_idx]

        save_output_npz(
            filename=os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
            arrs=concatenated_arrs,
            include_meta=include_meta,
            include_qvalues=include_qvalues,
            start=start,
            stop=stop,
        )

    return num_files_not_found

def write_one_output_file(
    filename: str,
    arrs: Sequence[np.ndarray | None],
    out_file_start: int,
    out_file_stop: int,
    include_meta: bool,
    include_qvalues: bool,
):
    """Write rows [out_file_start, out_file_stop) of arrs to one output npz + json metadata
    Returns the number of rows written.
    """
    num_rows = out_file_stop - out_file_start

    save_output_npz(
        filename=filename,
        arrs=arrs,
        include_meta=include_meta,
        include_qvalues=include_qvalues,
        start=out_file_start,
        stop=out_file_stop,
    )

    jsonfilename = os.path.splitext(filename)[0] + ".json"
    with open(jsonfilename,"w") as f:
        json.dump({"num_rows":num_rows},f)

    return num_rows

def merge_bucket(
    out_filenames: list[str],
    num_shards_to_merge: int,
    out_tmp_dir: str,
    include_meta: bool,
    include_qvalues: bool
):
    """Merge one shard bucket and split it across its assigned output files.

    Loads all shard files this bucket received (one per sharding group), concatenates and
    jointly shuffles them, then splits the shuffled rows into len(out_filenames) contiguous
    output files. Because shardify assigned rows to buckets by an independent uniform draw
    per row, and the bucket is uniformly permuted before splitting, each output file is a
    uniform random subset of all rows. The overall result is an exact uniform shuffle.

    Returns a list of (out_filename, num_rows_written) for the output files written.
    """
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(5)])

    binaryInputNCHWPackedList = []
    globalInputNCList = []
    policyTargetsNCMoveList = []
    globalTargetsNCList = []
    scoreDistrNList = []
    valueTargetsNCHWList = []
    metadataInputNCList = [] if include_meta else None
    qValueTargetsNCMoveList = [] if include_qvalues else None

    for input_idx in range(num_shards_to_merge):
        shard_filename = os.path.join(out_tmp_dir, str(input_idx) + ".npz")
        try:
            load_and_accumulate_input_contents(shard_filename,binaryInputNCHWPackedList,globalInputNCList,policyTargetsNCMoveList,globalTargetsNCList,scoreDistrNList,valueTargetsNCHWList,metadataInputNCList,qValueTargetsNCMoveList,include_meta,include_qvalues,fill_in_qvalues=False)
        except FileNotFoundError:
            print("WARNING: Empty shard in merge_bucket for shard :", input_idx, out_tmp_dir)

    if len(binaryInputNCHWPackedList) <= 0:
        print("WARNING: empty merge bucket: ", out_tmp_dir)
        return []

    concatenated_arrs = (
        joint_concatenate(
            [binaryInputNCHWPackedList,globalInputNCList,policyTargetsNCMoveList,globalTargetsNCList,scoreDistrNList,valueTargetsNCHWList,metadataInputNCList,qValueTargetsNCMoveList]
        )
    )

    assert_batch_dim_matches(concatenated_arrs)
    num_rows = concatenated_arrs[0].shape[0]

    concatenated_arrs = joint_shuffle_take_first_n(num_rows, concatenated_arrs)
    assert_batch_dim_matches(concatenated_arrs, num_rows)

    # print("%s: Merge writing... (mem usage %dMB)" % (str(datetime.datetime.now()),memusage_mb()), flush=True)

    # Split the shuffled bucket rows into the bucket's output files as evenly as possible.
    # The rows are already uniformly shuffled, so contiguous slices are uniform random subsets.
    num_out_files_here = len(out_filenames)
    assert num_out_files_here >= 1
    results = []
    for j, filename in enumerate(out_filenames):
        out_file_start = (j * num_rows) // num_out_files_here
        out_file_stop = ((j+1) * num_rows) // num_out_files_here
        rows_written = write_one_output_file(
            filename=filename,
            arrs=concatenated_arrs,
            out_file_start=out_file_start,
            out_file_stop=out_file_stop,
            include_meta=include_meta,
            include_qvalues=include_qvalues,
        )
        results.append((filename, rows_written))

    # Clean up scratch dir for this bucket now that we're done with it,
    # to reduce peak disk usage when there are many buckets.
    if os.path.exists(out_tmp_dir):
        shutil.rmtree(out_tmp_dir)

    return results

def group_files_by_rows(input_files_with_rows, worker_group_size):
    """Clump (filename,num_rows) pairs into worker groups of >= worker_group_size rows each.

    Returns a list of groups, where each group is a list of filenames. Files with
    num_rows <= 0 are skipped.
    """
    groups = []
    group_size_so_far = 0
    group_so_far = []
    for (input_file,num_rows_in_file) in input_files_with_rows:
        if num_rows_in_file <= 0:
            continue
        group_so_far.append(input_file)
        group_size_so_far += num_rows_in_file
        if group_size_so_far >= worker_group_size:
            groups.append(group_so_far)
            group_so_far = []
            group_size_so_far = 0
    if group_size_so_far > 0:
        groups.append(group_so_far)
    return groups

def compute_buckets_and_out_files(approx_rows, approx_rows_per_bucket, approx_rows_per_out_file):
    """Choose (num_buckets, num_out_files_per_bucket) for shuffling approx_rows rows."""
    # approx_rows_per_bucket is required (validated in main) to be a multiple of approx_rows_per_out_file
    assert approx_rows_per_bucket % approx_rows_per_out_file == 0
    num_buckets = max(int(round(approx_rows / approx_rows_per_bucket)), 1)
    num_out_files_per_bucket = approx_rows_per_bucket // approx_rows_per_out_file
    return num_buckets, num_out_files_per_bucket

def compute_desired_num_rows(num_usable_rows, min_rows, add_to_data_rows, taper_window_exponent,
                             expand_window_per_row, taper_window_scale, max_rows):
    """Compute the power-law window size (desired number of rows) from the run's usable rows.
    See the --help docstring for the meaning of the power-law parameters.
    Returns the desired number of rows, clamped to [min_rows, max_rows].
    """
    # How far offset do we start on the power-law window tail? Defaults to min_rows.
    window_taper_offset = taper_window_scale if taper_window_scale is not None else min_rows

    # Every post-random row moves one row beyond window_taper_offset.
    power_law_x = num_usable_rows - min_rows + window_taper_offset + add_to_data_rows
    # Apply power law and correct for window_taper_offset so we're still anchored at 0.
    unscaled_power_law = (power_law_x ** taper_window_exponent) - (window_taper_offset ** taper_window_exponent)
    # Scale so that we have an initial derivative of 1.
    scaled_power_law = unscaled_power_law / (taper_window_exponent * (window_taper_offset ** (taper_window_exponent-1)))
    # Scale so that we have the desired initial slope, and add back the minimum random rows.
    desired_num_rows = int(scaled_power_law * expand_window_per_row + min_rows)

    desired_num_rows = max(desired_num_rows, min_rows)
    if max_rows is not None:
        desired_num_rows = min(desired_num_rows, max_rows)
    return desired_num_rows

def format_bytes(num_bytes):
    """Format a byte count as a human-readable string."""
    v = float(num_bytes)
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if abs(v) < 1024.0 or unit == "PB":
            return "%.2f %s" % (v, unit)
        v /= 1024.0

def format_count(n):
    """Format a large integer count compactly (e.g. 1.2M)."""
    v = float(n)
    for unit in ["","K","M","G","T"]:
        if abs(v) < 1000.0 or unit == "T":
            return ("%.1f%s" % (v, unit)) if unit else ("%d" % int(v))
        v /= 1000.0

def estimate_shard_compressed_bytes_per_row(rows_per_shard_file, uncompressed_bytes_per_row):
    """Estimate compressed bytes/row for shard files holding rows_per_shard_file rows each."""
    asymptote = COMPRESSED_FRACTION_LARGE_FILE * uncompressed_bytes_per_row
    rows = max(1.0, float(rows_per_shard_file))
    return asymptote + CLIFF_OVERHEAD_BYTES_PER_FILE / rows

def print_resource_cost_estimate(
    label,
    final_input_rows,
    num_processes,
    worker_group_size,
    approx_rows_per_bucket,
    approx_rows_per_out_file,
    num_waves,
    include_meta,
    include_qvalues,
):
    """Print a rough estimate of the shuffle's resource costs for final_input_rows rows.

    final_input_rows is the number of rows that will actually be shuffled (after window
    selection, keep_prob subsampling, and md5 path filtering). All numbers assume 19x19
    data and use empirically-measured typical per-row sizes.
    """
    uncompressed_bytes_per_row = UNCOMPRESSED_BYTES_PER_ROW_REQUIRED_19
    if include_qvalues:
        uncompressed_bytes_per_row += UNCOMPRESSED_BYTES_PER_ROW_QVALUES_19
    if include_meta:
        uncompressed_bytes_per_row += UNCOMPRESSED_BYTES_PER_ROW_META_19

    rows_per_wave = final_input_rows / num_waves
    num_buckets, num_out_files_per_bucket = compute_buckets_and_out_files(
        rows_per_wave, approx_rows_per_bucket, approx_rows_per_out_file)

    # Sharding groups, per wave. Each group loads ~worker_group_size rows.
    num_groups_per_wave = max(1, int(round(rows_per_wave / worker_group_size)))
    # Intermediate shard files. Each two-phase shuffle writes num_groups * num_buckets shard
    # files. In wave mode there are ALSO the phase-1 scatter shards (num_phase1_groups * W
    # files); these are deleted one wave at a time as each wave is processed, but the peak is
    # during the first wave, when all phase-1 shards still exist alongside that wave's phase-2
    # shards. So the peak simultaneous count is the sum of the two (matching peak_disk_bytes).
    num_phase1_groups = max(1, int(round(final_input_rows / worker_group_size)))
    phase2_shard_files_per_wave = num_groups_per_wave * num_buckets
    if num_waves > 1:
        phase1_shard_files = num_phase1_groups * num_waves
        peak_shard_files = phase1_shard_files + phase2_shard_files_per_wave
    else:
        phase1_shard_files = 0
        peak_shard_files = phase2_shard_files_per_wave

    # Output files and rows.
    num_out_files = num_buckets * num_out_files_per_bucket * num_waves
    # The shuffle conserves rows, so output rows ~= final_input_rows.
    output_rows = final_input_rows

    # Peak intermediate disk usage. Shards hold ~rows_per_wave rows spread over
    # phase2_shard_files_per_wave files (phase 2) or final_input_rows over phase1_shard_files (phase 1).
    p2_rows_per_shard = rows_per_wave / max(1, phase2_shard_files_per_wave)
    p2_bytes = rows_per_wave * estimate_shard_compressed_bytes_per_row(p2_rows_per_shard, uncompressed_bytes_per_row)
    if num_waves > 1:
        p1_rows_per_shard = final_input_rows / max(1, phase1_shard_files)
        p1_bytes = final_input_rows * estimate_shard_compressed_bytes_per_row(p1_rows_per_shard, uncompressed_bytes_per_row)
        # Peak: phase-1 set fully written, draining as waves are processed, plus one wave's
        # phase-2 shards. A simple bound is p1_bytes + one wave's p2_bytes.
        peak_disk_bytes = p1_bytes + p2_bytes
    else:
        p1_rows_per_shard = 0
        peak_disk_bytes = p2_bytes

    # Peak memory.
    # Shardify worker: loads ~worker_group_size rows, concatenates, shuffles (a few copies).
    # Merge worker: loads one whole bucket (~approx_rows_per_bucket rows), concatenates and
    # shuffles. Use a small constant factor for the transient copies.
    shardify_copies = 3.0
    merge_copies = 3.0
    shardify_mem_per_worker = worker_group_size * uncompressed_bytes_per_row * shardify_copies
    merge_mem_per_worker = approx_rows_per_bucket * uncompressed_bytes_per_row * merge_copies
    shardify_mem_total = shardify_mem_per_worker * num_processes
    merge_mem_total = merge_mem_per_worker * num_processes

    # Final output dataset size. Output files hold ~approx_rows_per_out_file rows each
    output_compressed_bytes_per_row = estimate_shard_compressed_bytes_per_row(
        approx_rows_per_out_file, uncompressed_bytes_per_row)
    output_compressed_bytes = output_rows * output_compressed_bytes_per_row

    # Compressed bytes/row of the (smaller) intermediate shard files
    large_file_compressed_bytes_per_row = COMPRESSED_FRACTION_LARGE_FILE * uncompressed_bytes_per_row
    avg_shard_rows = (rows_per_wave / max(1, phase2_shard_files_per_wave))
    shard_compressed_bytes_per_row = estimate_shard_compressed_bytes_per_row(
        avg_shard_rows, uncompressed_bytes_per_row)

    print("==================================================================", flush=True)
    print("RESOURCE COST ESTIMATE (%s)" % label, flush=True)
    print("  ROUGH estimate, assumes 19x19 data and typical measured per-row sizes.", flush=True)
    print("  Final input rows to shuffle: %s" % format_count(final_input_rows), flush=True)
    print("  Per-row uncompressed size: %s" % format_bytes(uncompressed_bytes_per_row), flush=True)
    print("  Per-row final compressed size: %s" % format_bytes(output_compressed_bytes_per_row), flush=True)
    print("  Final compressed dataset size: %s" % format_bytes(output_compressed_bytes), flush=True)
    print("  Output: ~%s rows in ~%s files (%s buckets x %s files/bucket%s)" % (
        format_count(output_rows), format_count(num_out_files),
        format_count(num_buckets), format_count(num_out_files_per_bucket),
        (" x %d waves" % num_waves) if num_waves > 1 else ""), flush=True)
    if num_waves > 1:
        print("  Waves: %d (each ~%s rows, one live at a time)" % (num_waves, format_count(rows_per_wave)), flush=True)
        print("  Sharding groups: ~%s (phase 1), ~%s per wave (phase 2)" % (
            format_count(num_phase1_groups), format_count(num_groups_per_wave)), flush=True)
        print("  Peak intermediate shard files: ~%s (phase-1 %s + phase-2/wave %s)" % (
            format_count(peak_shard_files), format_count(phase1_shard_files),
            format_count(phase2_shard_files_per_wave)), flush=True)
    else:
        print("  Sharding groups: ~%s" % format_count(num_groups_per_wave), flush=True)
        print("  Peak intermediate shard files: ~%s (%s groups x %s buckets)" % (
            format_count(peak_shard_files), format_count(num_groups_per_wave), format_count(num_buckets)), flush=True)
    print("  Avg rows per shard file: ~%.1f, per row compressed size %s, ~%.1fx ideal (overly small shard files may compress less well)" % (
        avg_shard_rows,
        format_bytes(shard_compressed_bytes_per_row),
        shard_compressed_bytes_per_row / large_file_compressed_bytes_per_row), flush=True)
    print("  Peak intermediate temp disk: ~%s" % format_bytes(peak_disk_bytes), flush=True)
    print("  Peak memory, shardify: ~%s/worker x %d = ~%s" % (
        format_bytes(shardify_mem_per_worker), num_processes, format_bytes(shardify_mem_total)), flush=True)
    print("  Peak memory, merge:    ~%s/worker x %d = ~%s" % (
        format_bytes(merge_mem_per_worker), num_processes, format_bytes(merge_mem_total)), flush=True)
    print("==================================================================", flush=True)

def run_two_phase_shuffle(
    pool,
    label,
    input_files_with_rows,
    out_dir,
    out_file_prefix,
    num_buckets,
    num_out_files_per_bucket,
    bucket_tmp_dir,
    worker_group_size,
    keep_prob,
    include_meta,
    include_qvalues,
    fill_in_qvalues,
):
    """Run one shardify -> merge_bucket two-phase shuffle.

    Shardifies the input files into num_buckets buckets under bucket_tmp_dir (one bucket
    subdir each), then merges each bucket and splits it into num_out_files_per_bucket
    output files. Output files are named "{out_file_prefix}{bucket}_{idx}.npz" in out_dir,
    so each bucket numbers its own files independently (no global output index allocation).
    Returns a flat list of (out_file, num_rows_written).

    This is the core uniform shuffle used both for the whole dataset (non-wave mode) and
    for each individual wave (wave mode). fill_in_qvalues should be True when reading raw
    input npzs and False when re-sharding already-processed shard files (e.g. wave shards).

    The result is an exact uniform shuffle: each row's bucket is an independent uniform draw
    (shardify), and within a bucket rows are uniformly permuted before being split into
    output files, so each output file is a uniform random subset of the input.
    """
    num_buckets = max(1, num_buckets)
    num_out_files_per_bucket = max(1, num_out_files_per_bucket)

    bucket_tmp_dirs = [os.path.join(bucket_tmp_dir, "tmp.shuf%d" % b) for b in range(num_buckets)]
    for d in bucket_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    groups = group_files_by_rows(input_files_with_rows, worker_group_size)
    print("[%s] Grouping %d input files into %d sharding groups, %d buckets, %d output files/bucket" % (
        label, len(input_files_with_rows), len(groups), num_buckets, num_out_files_per_bucket), flush=True)

    if len(groups) <= 0:
        print("[%s] WARNING: no nonempty input files, nothing to shuffle" % label, flush=True)
        return []

    with TimeStuff("[%s] Sharding" % label):
        pool.starmap(shardify, [
            (group_idx, groups[group_idx], num_buckets, bucket_tmp_dirs, keep_prob,
             include_meta, include_qvalues, fill_in_qvalues)
            for group_idx in range(len(groups))
        ])

    with TimeStuff("[%s] Merging" % label):
        num_shards_to_merge = len(groups)
        merge_results = pool.starmap(merge_bucket, [
            (
                [
                    os.path.join(out_dir, "%s%d_%d.npz" % (out_file_prefix, b, j))
                    for j in range(num_out_files_per_bucket)
                ],
                num_shards_to_merge, bucket_tmp_dirs[b],
                include_meta, include_qvalues
            )
            for b in range(num_buckets)
        ])

    # merge_bucket cleans up its own bucket dir, but remove the parent shells too if empty.
    for d in bucket_tmp_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

    return [pair for bucket_result in merge_results for pair in bucket_result]

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
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(npyfile)
                elif version == (2, 0):
                    header = np.lib.format.read_array_header_2_0(npyfile)
                else:
                    raise NotImplementedError(f"Unexpected np version for {filename}: {version}")
                (shape, is_fortran, dtype) = header
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

    Additionally, NOT all of the shuffled window need be output: -keep-target-rows controls how many rows are randomly sampled and kept (pass 'all' to keep the whole window). For ongoing self-play training the intention is that this script is rerun as new data comes in, such that well before train.py would need more than -keep-target-rows rows, the data would have been reshuffled and a fresh random sample chosen.

    If you are NOT doing ongoing self-play training, but simply want to shuffle an entire dataset (not just a window of it) and output all of it, the default window args already select the whole dataset, so you just need:
      -keep-target-rows all

    If you ARE doing ongoing self-play training, but want a fixed window size, then you can use arguments like:
      -min-rows YOUR_DESIRED_SIZE \\
      -expand-window-per-row 0.0

    ==================================================================
    ALGORITHM:
    Compute total number of buckets B based on -approx-rows-per-bucket.

    * Shardify
       * Load files in chunks of -worker-group-size many data rows at a time, for a total of G groups.
       * Shuffle and distribute those rows among the buckets, in total writing G*B small shard files.
    * Merge
       * Load all the shards in each bucket, combine and shuffle and split the rows equally between the output files.

    When -num-waves W is provided, first does an extra shardify step, splitting among W waves (instead of B buckets).
    Then, each bucket is now treated as a separate dataset, and the whole algorithm (shardify + merge) is performed on each wave one by one by one, deleting the wave after it's done. This is because G*B can get quite large, for large datasets, especially if you choose parameters that make workers use less memory each, to allow for a lot of parallelism.

    You can preview a rough estimate of the resource costs (output files, peak intermediate shard count, peak temp disk usage, peak memory) without scanning the dataset by running with:
    --dry-run-print-resource-cost NUM_DATASET_ROWS
    which assumes the dataset has NUM_DATASET_ROWS total rows and prints estimates instead of shuffling.
    """)
    parser.add_argument('dirs', metavar='DIR', nargs='*', help='Directories of training data files (not required in --dry-run-print-resource-cost mode)')

    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    optional_args.add_argument('-min-rows', type=int, required=False, help='Minimum size of the desired training window, default 250k')
    optional_args.add_argument('-max-rows', type=int, required=False, help='Maximum size of the desired training window, default unbounded')
    required_args.add_argument('-keep-target-rows', required=True, help="Target number of rows to actually sample and keep in the final output shuffle, or 'all' to keep the whole window")
    optional_args.add_argument('-expand-window-per-row', type=float, required=False, default=1.0, help='Beyond min rows, initially expand the window by this much every post-random data row (default 1.0)')
    optional_args.add_argument('-taper-window-exponent', type=float, required=False, default=1.0, help='Make the window size asymtotically grow as this power of the data rows (default 1.0)')
    optional_args.add_argument('-taper-window-scale', type=float, required=False, help='The scale at which the power law applies, defaults to -min-rows')
    optional_args.add_argument('-add-to-data-rows', type=float, required=False, help='Compute the window size as if the number of data rows were this much larger/smaller')
    optional_args.add_argument('-summary-file', required=False, help='Summary json file for directory contents')
    optional_args.add_argument('-out-dir', required=False, help='Dir to output training files (not required in --dry-run-print-resource-cost mode)')
    optional_args.add_argument('-out-tmp-dir', required=False, help='Dir to use as scratch space (not required in --dry-run-print-resource-cost mode)')
    optional_args.add_argument('-approx-rows-per-out-file', type=int, required=False, default=70000, help='Number of rows per output file, default 70k')
    optional_args.add_argument('-approx-rows-per-bucket', type=int, required=False, help='Each merge worker takes one whole bucket in RAM and splits it equally into output files. Bigger buckets means shard files. Must be a multiple of -approx-rows-per-out-file. Default: equal to -approx-rows-per-out-file.')
    optional_args.add_argument('-num-waves', type=int, required=False, default=1, help='If > 1, shuffle in this many waves to bound peak intermediate shard count and temp disk usage for very large (whole-dataset) shuffles. Default 1 (no waves).')
    optional_args.add_argument('-dry-run-print-resource-cost', type=int, required=False, metavar='NUM_DATASET_ROWS', help='Do not actually shuffle (or even scan the dataset). Assume the dataset has this many total rows, run the window-size / keep / md5-filter math, and print rough estimates of output files, peak intermediate shard count, peak temp disk usage, and peak memory. Assumes 19x19 data and typical measured per-row sizes.')
    required_args.add_argument('-num-processes', type=int, required=True, help='Number of multiprocessing processes for shuffling in parallel')
    optional_args.add_argument('-worker-group-size', type=int, required=False, default=80000, help='Internally, target having many rows per parallel sharding worker (doesnt affect merge)')
    optional_args.add_argument('-exclude', required=False, help='Text file with npzs to ignore, one per line')
    optional_args.add_argument('-exclude-prefix', required=False, help='Prefix to concat to lines in exclude to produce the full file path')
    optional_args.add_argument('-exclude-basename', required=False, action="store_true", help='Consider an exclude to match if basename matches')
    optional_args.add_argument('-only-include-md5-path-prop-lbound', type=float, required=False, help='Just before sharding, include only filepaths hashing to float >= this')
    optional_args.add_argument('-only-include-md5-path-prop-ubound', type=float, required=False, help='Just before sharding, include only filepaths hashing to float < this')
    optional_args.add_argument('-skip-mtime-range-start', type=float, required=False, help='')
    optional_args.add_argument('-skip-mtime-range-end', type=float, required=False, help='')
    optional_args.add_argument('-include-meta', action="store_true", required=False, help='Include sgf metadata inputs')
    optional_args.add_argument('-exclude-qvalues', action="store_true", required=False, help='Exclude Q-value targets (for backwards compatibility with pre-v1.16)')

    args = parser.parse_args()
    dirs = args.dirs
    min_rows = args.min_rows
    max_rows = args.max_rows
    # -keep-target-rows is required, and accepts 'all' to mean "keep the whole window"
    # (represented internally as None, i.e. no cap).
    if str(args.keep_target_rows).lower() == "all":
        keep_target_rows = None
    else:
        keep_target_rows = int(args.keep_target_rows)
    expand_window_per_row = args.expand_window_per_row
    taper_window_exponent = args.taper_window_exponent
    taper_window_scale = args.taper_window_scale
    add_to_data_rows = args.add_to_data_rows

    summary_file = args.summary_file
    out_dir = args.out_dir
    out_tmp_dir = args.out_tmp_dir
    approx_rows_per_out_file = args.approx_rows_per_out_file
    approx_rows_per_bucket = args.approx_rows_per_bucket
    if approx_rows_per_bucket is None:
        approx_rows_per_bucket = approx_rows_per_out_file
    # Require buckets to split into a whole number of output files of the target size, so
    # each bucket maps cleanly onto num_out_files_per_bucket = bucket/out_file output files.
    if approx_rows_per_out_file <= 0:
        raise ValueError("-approx-rows-per-out-file (%d) must be positive" % approx_rows_per_out_file)
    if approx_rows_per_bucket <= 0:
        raise ValueError("-approx-rows-per-bucket (%d) must be positive" % approx_rows_per_bucket)
    if approx_rows_per_bucket % approx_rows_per_out_file != 0:
        raise ValueError(
            "-approx-rows-per-bucket (%d) must be a multiple of -approx-rows-per-out-file (%d)" % (
                approx_rows_per_bucket, approx_rows_per_out_file))
    num_waves = args.num_waves
    if num_waves < 1:
        raise ValueError("-num-waves must be >= 1")
    num_processes = args.num_processes
    worker_group_size = args.worker_group_size
    exclude = args.exclude
    exclude_prefix = args.exclude_prefix
    if exclude_prefix is None:
        exclude_prefix = ""
    exclude_basename = args.exclude_basename
    only_include_md5_path_prop_lbound = args.only_include_md5_path_prop_lbound
    only_include_md5_path_prop_ubound = args.only_include_md5_path_prop_ubound
    skip_mtime_range_start = args.skip_mtime_range_start
    skip_mtime_range_end = args.skip_mtime_range_end
    include_meta = args.include_meta
    include_qvalues = not args.exclude_qvalues
    dry_run_print_resource_cost = args.dry_run_print_resource_cost

    # dirs / out-dir / out-tmp-dir are only needed for a real run, not for the dry run.
    if dry_run_print_resource_cost is None:
        if len(dirs) <= 0:
            raise ValueError("At least one input directory is required (except in --dry-run-print-resource-cost mode)")
        if out_dir is None:
            raise ValueError("-out-dir is required (except in --dry-run-print-resource-cost mode)")
        if out_tmp_dir is None:
            raise ValueError("-out-tmp-dir is required (except in --dry-run-print-resource-cost mode)")

    if min_rows is None:
        print("NOTE: -min-rows was not specified, defaulting to requiring 250K rows before shuffling.")
        min_rows = 250000
    if add_to_data_rows is None:
        add_to_data_rows = 0

    # Fraction of files/rows kept by md5 path filtering, modeled as a uniform random fraction
    # of the dataset (since md5 hashes of filenames are ~uniform in [0,1)).
    md5_lbound = only_include_md5_path_prop_lbound if only_include_md5_path_prop_lbound is not None else 0.0
    md5_ubound = only_include_md5_path_prop_ubound if only_include_md5_path_prop_ubound is not None else 1.0
    md5_keep_fraction = max(0.0, min(1.0, md5_ubound) - max(0.0, md5_lbound))

    if dry_run_print_resource_cost is not None:
        # Estimate-only mode: do not scan the dataset at all. Assume the given total row
        # count and run the same window / keep / md5 math the real run would.
        num_dataset_rows = dry_run_print_resource_cost
        # Model usable rows as the whole dataset (we can't tell random vs post-random rows
        # without scanning, and the random-row cap only matters very early in a run).
        num_usable_rows = num_dataset_rows
        desired_num_rows = compute_desired_num_rows(
            num_usable_rows, min_rows, add_to_data_rows, taper_window_exponent,
            expand_window_per_row, taper_window_scale, max_rows)
        # Window selection takes the most recent desired_num_rows rows (capped by dataset).
        num_rows_used = min(desired_num_rows, num_dataset_rows)
        # keep_prob subsamples down toward keep_target_rows.
        approx_rows_to_keep = min(num_rows_used, keep_target_rows) if keep_target_rows is not None else num_rows_used
        keep_prob = approx_rows_to_keep / num_rows_used if num_rows_used > 0 else 0.0
        # md5 filtering removes a uniform fraction of the (windowed) rows, then keep_prob applies.
        num_rows_in_desired_files = num_rows_used * md5_keep_fraction
        final_input_rows = num_rows_in_desired_files * keep_prob

        print("DRY RUN: assuming dataset of %s rows (19x19), NOT scanning any files." % format_count(num_dataset_rows), flush=True)
        print("  Window: desired %s rows (taper_exp=%s, expand=%s, min_rows=%s, add_to_data_rows=%s%s)" % (
            format_count(desired_num_rows), taper_window_exponent, expand_window_per_row,
            format_count(min_rows), format_count(int(add_to_data_rows)),
            (", max_rows=%s" % format_count(max_rows)) if max_rows is not None else ""), flush=True)
        print("  Windowed rows used: %s; keep_prob=%.4f (toward keep_target_rows=%s)" % (
            format_count(num_rows_used), keep_prob,
            format_count(keep_target_rows) if keep_target_rows is not None else "none"), flush=True)
        if md5_keep_fraction < 1.0:
            print("  md5 path filter [%.3f, %.3f): keeping fraction %.4f" % (md5_lbound, md5_ubound, md5_keep_fraction), flush=True)
        print_resource_cost_estimate(
            label="dry run",
            final_input_rows=final_input_rows,
            num_processes=num_processes,
            worker_group_size=worker_group_size,
            approx_rows_per_bucket=approx_rows_per_bucket,
            approx_rows_per_out_file=approx_rows_per_out_file,
            num_waves=num_waves,
            include_meta=include_meta,
            include_qvalues=include_qvalues,
        )
        sys.exit(0)

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

    def num_usable_rows():
        global num_random_rows_capped
        global num_postrandom_rows
        return num_random_rows_capped + num_postrandom_rows

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
        if os.listdir(out_dir):
            raise Exception(out_dir + " already exists and is not empty")
    else:
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
    desired_num_rows = compute_desired_num_rows(
        num_usable_rows(), min_rows, add_to_data_rows, taper_window_exponent,
        expand_window_per_row, taper_window_scale, max_rows)
    print("Desired num rows: %d / %d" % (desired_num_rows,num_rows_total), flush=True)

    # add_to_data_rows accounts for data rows that are no longer present on disk (e.g. older data that was
    # deleted or archived), conceptually sitting at the oldest end of the dataset before any surviving file.
    # Offset the reported range by it so that the "range" output (used by train.py for logging/naming and the
    # train-bucket row accounting) reflects the full conceptual dataset rather than just the surviving rows.
    data_row_offset = int(add_to_data_rows)

    desired_input_files = []
    min_start_row = num_rows_total + data_row_offset
    max_end_row = num_rows_total + data_row_offset
    num_rows_used = 0
    print_stride = 1 + len(all_files) // 80
    end_row = num_rows_total + data_row_offset
    with TimeStuff("Computing desired rows"):
        for i in range(len(all_files)):
            (filename,mtime,num_rows) = all_files[i]

            # This could happen if the .summary.json file is inaccurate after file deletions
            # Actually we just handle that in shardify - and accept that it might make our window slightly not far back enough
            # if not os.path.exists(filename):
            #   continue

            if skip_mtime_range_start is not None and skip_mtime_range_end is not None:
                if mtime >= skip_mtime_range_start and mtime <= skip_mtime_range_end:
                    if np.random.randint(100000) == 0:
                        print("DEBUG: skip mtime " + filename, flush=True)
                    continue

            if num_rows is not None and num_rows > 0:
                desired_input_files.append((filename,num_rows))
                start_row = end_row - num_rows
                min_start_row = min(start_row, min_start_row)
                num_rows_used += num_rows
            else:
                start_row = end_row

            if i % print_stride == 0 or num_rows_used >= desired_num_rows:
                print("Using: %s (%d-%d) (%d/%d desired rows)" % (filename,start_row,end_row,num_rows_used,desired_num_rows), flush=True)
            if num_rows_used >= desired_num_rows:
                break

            # Update end row for next loop
            end_row = start_row

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

    approx_rows_to_keep_after_md5 = num_rows_in_desired_files * keep_prob

    # Informationally print the resource-cost estimate for the actual computed window, using
    # the same estimator as the dry run. These are rough estimates; the shuffle proceeds.
    print_resource_cost_estimate(
        label="estimate for this run",
        final_input_rows=approx_rows_to_keep_after_md5,
        num_processes=num_processes,
        worker_group_size=worker_group_size,
        approx_rows_per_bucket=approx_rows_per_bucket,
        approx_rows_per_out_file=approx_rows_per_out_file,
        num_waves=num_waves,
        include_meta=include_meta,
        include_qvalues=include_qvalues,
    )

    if num_waves <= 1:
        # Single-pass (no waves): one two-phase shuffle over the whole dataset.
        num_buckets, num_out_files_per_bucket = compute_buckets_and_out_files(
            approx_rows_to_keep_after_md5, approx_rows_per_bucket, approx_rows_per_out_file)
        print("Writing ~%d output files (%d buckets x %d files/bucket), approx %d kept rows (keep_prob %.4f of %d desired rows, %d pre-md5 rows)" % (
            num_buckets * num_out_files_per_bucket, num_buckets, num_out_files_per_bucket,
            approx_rows_to_keep_after_md5, keep_prob, num_rows_in_desired_files, desired_num_rows), flush=True)

        bucket_tmp_dir = os.path.join(out_tmp_dir, "buckets")
        if os.path.exists(bucket_tmp_dir):
            shutil.rmtree(bucket_tmp_dir)
        os.makedirs(bucket_tmp_dir, exist_ok=True)

        with multiprocessing.Pool(num_processes) as pool:
            all_written = run_two_phase_shuffle(
                pool=pool,
                label="whole",
                input_files_with_rows=desired_input_files,
                out_dir=out_dir,
                out_file_prefix="data",
                num_buckets=num_buckets,
                num_out_files_per_bucket=num_out_files_per_bucket,
                bucket_tmp_dir=bucket_tmp_dir,
                worker_group_size=worker_group_size,
                keep_prob=keep_prob,
                include_meta=include_meta,
                include_qvalues=include_qvalues,
                fill_in_qvalues=True,
            )
        print("Number of rows by output file:", flush=True)
        print(all_written, flush=True)
        sys.stdout.flush()

        if os.path.exists(bucket_tmp_dir):
            shutil.rmtree(bucket_tmp_dir)
    else:
        # Wave mode bounds peak intermediate shard count and temp disk usage when a single
        # bucket must fit one merge worker's RAM but num_sharding_groups * num_buckets would
        # still be huge for a whole-dataset shuffle. Phase 1 scatters every row uniformly at
        # random into one of W waves (per row, not per-file, this makes the shuffle perfect)
        # Each wave is then shuffled independently by the ordinary two-phase
        # shardify+merge, and its temporaries are deleted before the next wave, so only one
        # wave's shards exist on disk at a time. Merge-worker memory is unchanged.
        approx_kept_rows_per_wave = approx_rows_to_keep_after_md5 / num_waves
        num_buckets_per_wave, num_out_files_per_bucket = compute_buckets_and_out_files(
            approx_kept_rows_per_wave, approx_rows_per_bucket, approx_rows_per_out_file)
        print("Wave mode: %d waves, approx %d kept rows total (~%d rows/wave), %d buckets/wave x %d files/bucket" % (
            num_waves, approx_rows_to_keep_after_md5, approx_kept_rows_per_wave,
            num_buckets_per_wave, num_out_files_per_bucket), flush=True)

        wave_tmp_root = os.path.join(out_tmp_dir, "waves")
        if os.path.exists(wave_tmp_root):
            shutil.rmtree(wave_tmp_root)
        wave_dirs = [os.path.join(wave_tmp_root, "wave%d" % w) for w in range(num_waves)]
        for d in wave_dirs:
            os.makedirs(d, exist_ok=True)

        bucket_tmp_dir = os.path.join(out_tmp_dir, "buckets")
        if os.path.exists(bucket_tmp_dir):
            shutil.rmtree(bucket_tmp_dir)
        os.makedirs(bucket_tmp_dir, exist_ok=True)

        # Group the original input files for phase 1 sharding.
        phase1_groups = group_files_by_rows(desired_input_files, worker_group_size)
        print("Phase 1: grouping %d input files into %d sharding groups, scattering into %d waves" % (
            len(desired_input_files), len(phase1_groups), num_waves), flush=True)

        all_written = []
        with multiprocessing.Pool(num_processes) as pool:
            # ---- Phase 1: scatter every row uniformly at random into one of W waves. ----
            # Reuses shardify exactly: "output files" here are the W wave dirs, so a row's
            # wave is an independent uniform draw. keep_prob subsampling happens here (once).
            with TimeStuff("Phase 1 scatter into waves"):
                pool.starmap(shardify, [
                    (group_idx, phase1_groups[group_idx], num_waves, wave_dirs, keep_prob,
                     include_meta, include_qvalues, True)
                    for group_idx in range(len(phase1_groups))
                ])

            # ---- Phase 2: shuffle each wave independently, then delete it. ----
            num_phase1_shards = len(phase1_groups)
            for w in range(num_waves):
                # This wave's "input files" are the phase-1 shards that landed in it.
                wave_shard_files = [
                    os.path.join(wave_dirs[w], "%d.npz" % group_idx)
                    for group_idx in range(num_phase1_shards)
                    if os.path.exists(os.path.join(wave_dirs[w], "%d.npz" % group_idx))
                ]
                # Get row counts so phase-2 worker-grouping is balanced. Cheap header reads.
                wave_input_files_with_rows = []
                for f in wave_shard_files:
                    (_, nr) = compute_num_rows(f)
                    wave_input_files_with_rows.append((f, nr if nr is not None else 0))

                wave_written = run_two_phase_shuffle(
                    pool=pool,
                    label="wave %d/%d" % (w, num_waves),
                    input_files_with_rows=wave_input_files_with_rows,
                    out_dir=out_dir,
                    out_file_prefix="data%d_" % w,
                    num_buckets=num_buckets_per_wave,
                    num_out_files_per_bucket=num_out_files_per_bucket,
                    bucket_tmp_dir=bucket_tmp_dir,
                    worker_group_size=worker_group_size,
                    keep_prob=1.0,  # keep_prob already applied in phase 1
                    include_meta=include_meta,
                    include_qvalues=include_qvalues,
                    fill_in_qvalues=False,  # wave shards already contain qValueTargetsNCMove
                )
                all_written.extend(wave_written)

                # Incrementally delete this wave's shards now that it's fully shuffled,
                # to free disk space (these are temporaries, not the original dataset).
                if os.path.exists(wave_dirs[w]):
                    shutil.rmtree(wave_dirs[w])

        print("Number of rows by output file:", flush=True)
        print(all_written, flush=True)
        sys.stdout.flush()

        if os.path.exists(wave_tmp_root):
            shutil.rmtree(wave_tmp_root)
        if os.path.exists(bucket_tmp_dir):
            shutil.rmtree(bucket_tmp_dir)

    dump_value = {
        "range": (min_start_row, max_end_row)
    }

    with open(out_dir + ".json", 'w') as f:
        json.dump(dump_value, f)
