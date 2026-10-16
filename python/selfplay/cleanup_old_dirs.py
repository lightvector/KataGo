#!/usr/bin/env python3
"""
Removes old subdirectories, keeping the most recent 3 that are older than 2 hours.
"""

import os
import shutil
import sys
import time

def main():
    directory = sys.argv[1]
    time_threshold = time.time() - 2 * 3600  # 2 hours in seconds
    dirs = []
    with os.scandir(directory) as it:
        for entry in it:
            try:
                s = entry.stat(follow_symlinks=False)
                if entry.is_dir(follow_symlinks=False) and s.st_mtime < time_threshold:
                    dirs.append((s.st_mtime, entry.path))
            except OSError as e:
                print(f"Error accessing {repr(entry.path)}: {e}", file=sys.stderr)
    dirs.sort()
    for (_, path) in dirs[:-3]:
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
