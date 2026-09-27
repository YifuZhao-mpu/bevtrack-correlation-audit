#!/usr/bin/env bash
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1; QLOG=wacv_queue.log
[ "$GPU" = "0" ] && cfg=official_base_s2 || cfg=official_gated_s2
echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (official-s2)" >> "$QLOG"
CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$? (official-s2)" >> "$QLOG"
