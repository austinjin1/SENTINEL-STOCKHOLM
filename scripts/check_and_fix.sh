#!/bin/bash
# Free /tmp space and check training status
# Output goes to /home/bcheng/SENTINEL/status.txt

exec > /home/bcheng/SENTINEL/status.txt 2>&1

echo "=== Cleaning /tmp ==="
find /tmp/claude-12189 -name "*.output" -not -name "bv2isdytk.output" -delete 2>/dev/null
echo "Cleaned old output files"

echo "=== Disk ==="
df -h /tmp

echo "=== Process check ==="
pgrep -af "train_aquassm_final" || echo "NOT RUNNING"

echo "=== Checkpoint files ==="
ls -la /home/bcheng/SENTINEL/checkpoints/sensor/aquassm_final_best.pt 2>/dev/null
ls -la /home/bcheng/SENTINEL/checkpoints/sensor/results_final.json 2>/dev/null

echo "=== Results ==="
cat /home/bcheng/SENTINEL/checkpoints/sensor/results_final.json 2>/dev/null || echo "No results file"

echo "=== Done ==="
