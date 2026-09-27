#!/usr/bin/env bash
# Stage 3: third seeds for evidence (GPU1) / gated (GPU0), if triggered.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
GPU=$1
QLOG=wacv_queue.log
if [ "$GPU" = "0" ]; then QUEUE=(fix_gated_s2); else QUEUE=(fix_evid_s2); fi
for cfg in "${QUEUE[@]}"; do
    echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (stage3)" >> "$QLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$? (stage3)" >> "$QLOG"
done
