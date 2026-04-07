#!/bin/bash
# Wait for NWIS+S2 download to complete, then retrain HydroViT on GPU 1

DOWNLOAD_PID=1824668
LOG=logs/hydrovit_v3_train.log
RESULT=checkpoints/satellite/hydrovit_wq_v3_results.json

echo "[$(date)] Waiting for download PID $DOWNLOAD_PID..."

while kill -0 $DOWNLOAD_PID 2>/dev/null; do
    sleep 60
    n=$(ls data/processed/satellite/nwis_tiles/*.npz 2>/dev/null | wc -l)
    echo "[$(date)] Download running, $n tiles cached"
done

echo "[$(date)] Download complete. Starting HydroViT v3 training..."
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_hydrovit_wq.py 2>&1 | tee $LOG
echo "[$(date)] Training complete."
