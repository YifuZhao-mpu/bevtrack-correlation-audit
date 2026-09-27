#!/usr/bin/env bash
# Van + Cyclist category sweep (journal/PR route: cross-category pooled analysis).
# GPU0: van base/forced/gated/evid s0 + cyc s0 ; GPU1: same for s1.
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1; S=$2; QLOG=wacv_queue.log
for cat in van cyc; do
  for arm in base forced gated evid; do
    cfg="fix_${cat}_${arm}_s${S}"
    echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (catsweep)" >> "$QLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$? (catsweep)" >> "$QLOG"
  done
done
echo "$(date '+%m/%d %H:%M') GPU$GPU catsweep seed$S done" >> "$QLOG"
