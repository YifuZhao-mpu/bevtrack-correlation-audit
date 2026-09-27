#!/usr/bin/env bash
# Third seed for ped/van/cyc (separates seed variance from category heterogeneity).
# GPU0: ped 4 arms ; GPU1: van 4 arms then cyc 4 arms (van/cyc are fast).
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
GPU=$1; QLOG=wacv_queue.log
[ "$GPU" = "0" ] && CATS=(ped) || CATS=(van cyc)
for cat in "${CATS[@]}"; do for arm in base forced gated evid; do
  cfg="fix_${cat}_${arm}_s2"
  echo "$(date '+%m/%d %H:%M') GPU$GPU launching $cfg (seed2)" >> "$QLOG"
  CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" > "${cfg}_car.log" 2>&1
  rc=$?; echo "$(date '+%m/%d %H:%M') GPU$GPU $cfg exit=$rc (seed2)" >> "$QLOG"
  if [ $rc -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/eval_tracklets.py "configs/kitti/$cfg.py" \
      "work_dir/$cfg/epoch_500.pth" --out "test_evals/$cfg.json" > "test_evals/$cfg.log" 2>&1
  fi
done; done
echo "$(date '+%m/%d %H:%M') GPU$GPU seed2 sweep done" >> "$QLOG"
