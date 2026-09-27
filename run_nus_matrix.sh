#!/usr/bin/env bash
# nuScenes matrix (PR package req#1: untouched MULTI-SCENE data; req#4: 5 paired seeds).
# 4 matched arms x 5 seeds = 20 runs, ~4h each. GPU0 takes seeds 0,2,4 ; GPU1 seeds 1,3.
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1; QLOG=wacv_queue.log
[ "$GPU" = "0" ] && SEEDS=(0 2 4) || SEEDS=(1 3)
for s in "${SEEDS[@]}"; do for arm in base forced gated evid; do
  cfg="nus_${arm}_s${s}"
  echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (nus)" >> "$QLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/nuscenes/$cfg.py" > "${cfg}.log" 2>&1
  rc=$?; echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$rc (nus)" >> "$QLOG"
  if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/eval_tracklets.py "configs/nuscenes/$cfg.py" \
      "work_dir/$cfg/epoch_40.pth" --out "test_evals/$cfg.json" > "test_evals/$cfg.log" 2>&1
    echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg eval done (nus)" >> "$QLOG"
  fi
done; done
echo "$(date '+%m/%d %H:%M') GPU$GPU nus matrix done" >> "$QLOG"
