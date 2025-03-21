#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import time
import logging
import zipfile
import json
import datetime
import dateutil.parser
import shutil

import multiprocessing

import numpy as np

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

def is_temp_npz_like(filename):
    return "_" in filename

def summarize_dir(dirpath):
    filenames = [filename for filename in os.listdir(dirpath) if filename.endswith('.npz')]

    num_rows_this_dir = 0
    filename_mtime_num_rowss = []
    for filename in filenames:
        filepath = os.path.join(dirpath,filename)
        mtime = os.path.getmtime(filepath)

        # Files that look like they are temp files should be recorded and warned
        if is_temp_npz_like(filename):
            print("WARNING: file looks like a temp file: ", filepath)
            filename_mtime_num_rowss.append((filename,mtime,None))
            continue

        try:
            npheaders = get_numpy_npz_headers(filepath)
        except PermissionError:
            print("WARNING: No permissions for reading file: ", filepath)
            filename_mtime_num_rowss.append((filename,mtime,None))
            continue
        except zipfile.BadZipFile:
            print("WARNING: Bad zip file: ", filepath)
            filename_mtime_num_rowss.append((filename,mtime,None))
            continue

        if npheaders is None or len(npheaders) <= 0:
            print("WARNING: bad npz headers for file: ", filepath)
            filename_mtime_num_rowss.append((filename,mtime,None))
            continue

        if "binaryInputNCHWPacked" in npheaders:
            (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
        else:
            (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked.npy"]
        num_rows = shape[0]
        num_rows_this_dir += num_rows

        filename_mtime_num_rowss.append((filename,mtime,num_rows))

    print("Summarizing new dir with %d rows: %s" % (num_rows_this_dir,dirpath),flush=True)
    return (dirpath, filename_mtime_num_rowss, num_rows_this_dir)


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
    parser = argparse.ArgumentParser(description='Shuffle data files')
    parser.add_argument('dirs', metavar='DIR', nargs='+', help='Directories of training data files')
    parser.add_argument('-old-summary-file-to-assume-correct', required=False, help='Summary json file for directory contents')
    parser.add_argument('-new-summary-file', required=True, help='Summary json file for directory contents')
    parser.add_argument('-num-parallel-processes', required=False, type=int, help='Number of parallel processes to use, default 4')

    args = parser.parse_args()
    dirs = args.dirs
    old_summary_file_to_assume_correct = args.old_summary_file_to_assume_correct
    new_summary_file = args.new_summary_file

    num_processes = 4
    if args.num_parallel_processes is not None:
        num_processes = args.num_parallel_processes

    summary_data_by_dirpath = {}
    if old_summary_file_to_assume_correct is not None and os.path.exists(old_summary_file_to_assume_correct):
        with TimeStuff("Loading " + old_summary_file_to_assume_correct):
            with open(old_summary_file_to_assume_correct) as fp:
                summary_data_by_dirpath = json.load(fp)

    # Update old summary data into new format
    dirpaths = list(summary_data_by_dirpath.keys())
    format_was_updated = False
    for dirpath in dirpaths:
        if "dir_mtime" not in summary_data_by_dirpath[dirpath]:
            format_was_updated = True
            filename_mtime_num_rowss = summary_data_by_dirpath[dirpath]
            summary_data_by_dirpath[dirpath] = {
                "dir_mtime": os.path.getmtime(dirpath),
                "filename_mtime_num_rowss": filename_mtime_num_rowss,
            }

    dirs_to_handle = []
    with TimeStuff("Finding files"):
        for d in dirs:
            for (path,dirnames,filenames) in os.walk(d, followlinks=True):
                had_no_dirnames = len(dirnames) == 0
                i = 0
                while i < len(dirnames):
                    dirname = dirnames[i]
                    dirpath = os.path.normpath(os.path.join(path, dirname))
                    if dirpath in summary_data_by_dirpath:
                        if os.path.getmtime(dirpath) == summary_data_by_dirpath[dirpath]["dir_mtime"]:
                            # Skip
                            del dirnames[i]
                            continue

                    if dirname == "tdata":
                        # Handle this dir and don't recurse further
                        del dirnames[i]
                        dirs_to_handle.append(dirpath)
                        continue

                    else:
                        parseddate = None
                        try:
                            parseddate = dateutil.parser.parse(dirname)
                        except ValueError:
                            parseddate = None
                        if parseddate is not None and parseddate < datetime.datetime.now() - datetime.timedelta(days=2.0):
                            # Handle this dir and don't recurse further
                            del dirnames[i]
                            dirs_to_handle.append(dirpath)
                            continue

                    i += 1

    with TimeStuff("Parallel summarizing %d dirs" % len(dirs_to_handle)):
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(summarize_dir,dirs_to_handle)

    num_total_rows = 0
    with TimeStuff("Merging %d results" % len(results)):
        for result in results:
            if result is None:
                continue
            (dirpath, filename_mtime_num_rowss, num_rows_this_dir) = result
            num_total_rows += num_rows_this_dir
            summary_data_by_dirpath[os.path.abspath(dirpath)] = {
                "dir_mtime": os.path.getmtime(os.path.abspath(dirpath)),
                "filename_mtime_num_rowss": filename_mtime_num_rowss,
            }

    if len(dirs_to_handle) == 0 and old_summary_file_to_assume_correct is not None and os.path.exists(old_summary_file_to_assume_correct) and not format_was_updated:
        shutil.copy(old_summary_file_to_assume_correct,new_summary_file)
        print("Not writing any new summary, no results, just copying old file")
    else:
        with TimeStuff("Writing result"):
            with open(new_summary_file,"w") as fp:
                json.dump(summary_data_by_dirpath,fp)
        print("Summary file written adding %d additional rows: %s" % (num_total_rows,new_summary_file),flush=True)

    print("Done computing new summary",flush=True)
    sys.stdout.flush()
