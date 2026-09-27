#!/usr/bin/env bash
# Stage 4: official-architecture confirmation (base vs gated, GPU-swapped seeds).
# GPU0: official_base_s0 -> official_gated_s1 ; GPU1: official_gated_s0 -> official_base_s1
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
GPU=$1
QLOG=wacv_queue.log
if [ "$GPU" = "0" ]; then QUEUE=(official_base_s0 official_gated_s1); else QUEUE=(official_gated_s0 official_base_s1); fi
for cfg in "${QUEUE[@]}"; do
    echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (stage4)" >> "$QLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
    rc=$?
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$rc (stage4)" >> "$QLOG"
    [ $rc -ne 0 ] && { echo "$(date '+%m/%d %H:%M') GPU$GPU stage4 ABORTED at $cfg" >> "$QLOG"; exit $rc; }
done
echo "$(date '+%m/%d %H:%M') GPU$GPU stage4 done" >> "$QLOG"
