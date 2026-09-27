#!/usr/bin/env bash
# Corrected 2-seed matrix (RESEARCH_REVIEW_2026-07-18 round 2, Q2 schedule).
# GPU0: base_s0 -> forced_s1 -> gated_s0 -> gated_s1
# GPU1: forced_s0 -> base_s1 -> evid_s0  -> evid_s1
# (GPU assignment swaps between baseline/forced seeds to detect machine effects.)
# Usage: bash run_wacv_matrix.sh <gpu_id>
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
GPU=$1
QLOG=wacv_queue.log

if [ "$GPU" = "0" ]; then
    QUEUE=(fix_base_s0 fix_forced_s1 fix_gated_s0 fix_gated_s1)
else
    QUEUE=(fix_forced_s0 fix_base_s1 fix_evid_s0 fix_evid_s1)
fi

for cfg in "${QUEUE[@]}"; do
    echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg" >> "$QLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
    rc=$?
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$rc" >> "$QLOG"
    if [ $rc -ne 0 ]; then
        echo "$(date '+%m/%d %H:%M') GPU$GPU queue ABORTED at $cfg" >> "$QLOG"
        exit $rc
    fi
done
echo "$(date '+%m/%d %H:%M') GPU$GPU queue done" >> "$QLOG"
