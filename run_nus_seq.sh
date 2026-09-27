#!/usr/bin/env bash
# nuScenes matrix, STRICTLY SEQUENTIAL on GPU0 (9.7GB preload each; two concurrent OOM'd).
set -u; PY=/home/zyf/anaconda3/envs/se2track/bin/python; cd /home/zyf/paper/BEVTrack
QLOG=wacv_queue.log
for s in 0 1 2 3 4; do for arm in base forced gated evid; do
  cfg="nus_${arm}_s${s}"
  [ -f "test_evals/$cfg.json" ] && continue
  echo "$(date '+%m/%d %H:%M') GPU0 launching $cfg (nus-seq)" >> "$QLOG"
  CUDA_VISIBLE_DEVICES=0 $PY train.py "configs/nuscenes/$cfg.py" > "${cfg}.log" 2>&1
  rc=$?; echo "$(date '+%m/%d %H:%M') GPU0 $cfg exit=$rc (nus-seq)" >> "$QLOG"
  [ $rc -eq 0 ] && CUDA_VISIBLE_DEVICES=0 $PY tools/eval_tracklets.py "configs/nuscenes/$cfg.py" \
      "work_dir/$cfg/epoch_40.pth" --out "test_evals/$cfg.json" > "test_evals/$cfg.log" 2>&1
done; done
echo "$(date '+%m/%d %H:%M') GPU0 nus matrix done" >> "$QLOG"
