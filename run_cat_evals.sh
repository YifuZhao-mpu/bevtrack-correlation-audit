#!/usr/bin/env bash
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1; S=$2
for cat in van cyc; do for arm in base forced gated evid; do
  run="fix_${cat}_${arm}_s${S}"
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/eval_tracklets.py configs/kitti/$run.py \
    work_dir/$run/epoch_500.pth --out test_evals/$run.json > test_evals/$run.log 2>&1
done; done
echo "$(date '+%m/%d %H:%M') GPU$GPU cat Test evals seed$S done" >> wacv_queue.log
