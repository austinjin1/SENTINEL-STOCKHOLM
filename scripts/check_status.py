#!/usr/bin/env python3
"""Check training status."""
import os, json, subprocess

# Check if training is running
result = subprocess.run(['pgrep', '-f', 'train_aquassm_final'], capture_output=True, text=True)
print(f"Running PIDs: {result.stdout.strip() or 'NONE'}")

# Check checkpoint mtime
for f in ['aquassm_final_best.pt', 'results_final.json']:
    fp = f'checkpoints/sensor/{f}'
    if os.path.exists(fp):
        mtime = os.path.getmtime(fp)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        sz = os.path.getsize(fp)
        print(f"{f}: mtime={dt}, size={sz}")

# Check results
fp = 'checkpoints/sensor/results_final.json'
if os.path.exists(fp):
    with open(fp) as fh:
        data = json.load(fh)
    print(f"Results: {json.dumps(data, indent=2)[:500]}")

# Check disk
result2 = subprocess.run(['df', '-h', '/tmp'], capture_output=True, text=True)
print(f"Disk: {result2.stdout}")
