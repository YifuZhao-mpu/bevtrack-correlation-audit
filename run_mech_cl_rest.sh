#!/usr/bin/env bash
# Remaining closed-loop mech seeds (run after Ped frees GPUs). GPU arg picks the half.
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1
if [ "$GPU" = "0" ]; then
  JOBS=("fix_gated_s1:surrogate_s1" "fix_gated_s2:surrogate_s2")
else
  JOBS=("official_gated_s1:official_s1" "official_gated_s2:official_s2")
fi
for j in "${JOBS[@]}"; do
  cfg="${j%%:*}"; tag="${j##*:}"
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/mech_closedloop.py --config configs/kitti/$cfg.py \
    --ckpt work_dir/$cfg/epoch_500.pth --split Test --tag $tag --out mech_cl/ > mech_cl/$tag.log 2>&1
  echo "$(date '+%m/%d %H:%M') mech_cl $tag done" >> wacv_queue.log
done
