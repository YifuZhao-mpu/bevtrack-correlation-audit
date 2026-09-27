#!/usr/bin/env bash
# Second-category (Pedestrian) corrected matrix, round-3 gate 6. stride_m=0.16 baked into configs.
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1; QLOG=wacv_queue.log
if [ "$GPU" = "1" ]; then QUEUE=(fix_ped_base_s0 fix_ped_forced_s0 fix_ped_gated_s0 fix_ped_evid_s0)
else QUEUE=(fix_ped_base_s1 fix_ped_forced_s1 fix_ped_gated_s1 fix_ped_evid_s1); fi
for cfg in "${QUEUE[@]}"; do
    echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (ped)" >> "$QLOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$? (ped)" >> "$QLOG"
done
echo "$(date '+%m/%d %H:%M') GPU$GPU ped queue done" >> "$QLOG"
