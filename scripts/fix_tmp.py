#!/usr/bin/env python3
"""Fix /tmp space by truncating old output files."""
import os, glob

# Find and truncate all .output files in the claude tmp dir
pattern = "/tmp/claude-12189/-home-bcheng-SENTINEL/*/tasks/*.output"
files = glob.glob(pattern)
freed = 0
for f in files:
    if 'bv2isdytk' in f:
        continue  # Keep the main training output
    try:
        sz = os.path.getsize(f)
        if sz > 0:
            open(f, 'w').close()  # truncate
            freed += sz
    except:
        try:
            os.unlink(f)
            freed += sz
        except:
            pass

print(f"Freed {freed} bytes from {len(files)} files")
