#!/usr/bin/env bash
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
for j in "fix_gated_s0:surrogate_v2_s0" "official_gated_s0:official_v2_s0"; do
  cfg="${j%%:*}"; tag="${j##*:}"
  CUDA_VISIBLE_DEVICES=1 $PY tools/mech_closedloop.py --config configs/kitti/$cfg.py \
    --ckpt work_dir/$cfg/epoch_500.pth --split Test --tag $tag --out mech_cl/ > mech_cl/$tag.log 2>&1
  echo "$(date '+%m/%d %H:%M') mech_v2 $tag done" >> wacv_queue.log
done
