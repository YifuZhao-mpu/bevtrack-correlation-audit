#!/usr/bin/env bash
# Stage 2: third seeds for base/forced (trigger: sign disagreement at 2 seeds).
# Usage: bash run_wacv_stage2.sh <gpu_id>  (GPU0: base_s2; GPU1: forced_s2)
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
GPU=$1
QLOG=wacv_queue.log
if [ "$GPU" = "0" ]; then QUEUE=(fix_base_s2); else QUEUE=(fix_forced_s2); fi
for cfg in "${QUEUE[@]}"; do
    echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (stage2)" >> "$QLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$? (stage2)" >> "$QLOG"
done
