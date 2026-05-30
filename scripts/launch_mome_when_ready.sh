#!/bin/bash
# Waits for Stream GNN to finish, then launches MoME fusion training
LOG=/home/bcheng/SENTINEL/logs/train_stream_gnn.log
while true; do
    if ! pgrep -f "train_stream_gnn" > /dev/null 2>&1; then
        echo "$(date): Stream GNN finished. Launching MoME fusion..."
        cd /home/bcheng/SENTINEL
        mkdir -p logs
        PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2 nohup conda run --no-capture-output -n physiformer \
            python scripts/train_mome_fusion.py --epochs 100 \
            > logs/train_mome_fusion.log 2>&1 &
        echo "MoME fusion launched with PID $!"
        break
    fi
    sleep 30
done
