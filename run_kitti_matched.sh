#!/usr/bin/env bash
# KITTI Car matched-head matrix (4 arms x 3 seeds), GPU1. ~5.2h/run, low RAM (~5GB).
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
QLOG=wacv_queue.log
for s in 0 1 2; do for arm in base forced gated evid; do
  cfg="m_${arm}_s${s}"
  echo "$(date '+%m/%d %H:%M') GPU1 launching $cfg (matched)" >> "$QLOG"
  CUDA_VISIBLE_DEVICES=1 $PY train.py "configs/kitti/$cfg.py" > "${cfg}.log" 2>&1
  rc=$?; echo "$(date '+%m/%d %H:%M') GPU1 $cfg exit=$rc (matched)" >> "$QLOG"
  [ $rc -eq 0 ] && CUDA_VISIBLE_DEVICES=1 $PY tools/eval_tracklets.py "configs/kitti/$cfg.py" \
      "work_dir/$cfg/epoch_500.pth" --out "test_evals/$cfg.json" > "test_evals/$cfg.log" 2>&1
done; done
echo "$(date '+%m/%d %H:%M') GPU1 matched matrix done" >> "$QLOG"
