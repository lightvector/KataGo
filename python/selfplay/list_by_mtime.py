#!/usr/bin/env python3
"""
Lists entries in a directory sorted by modification time, one path per line.
"""

import os
import sys

def main():
    directory = sys.argv[1]
    entries = []
    with os.scandir(directory) as it:
        for entry in it:
            try:
                entries.append((entry.stat(follow_symlinks=False).st_mtime, entry.path))
            except OSError as e:
                print(f"Error accessing {repr(entry.path)}: {e}", file=sys.stderr)
    entries.sort()
    if entries:
        print("\n".join(path for _, path in entries))


if __name__ == "__main__":
    main()
