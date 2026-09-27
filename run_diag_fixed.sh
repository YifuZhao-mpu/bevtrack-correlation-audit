#!/usr/bin/env bash
# Rerun the correlation diagnostic with the corrected query-relative decode and the
# true 0.60 m feature stride, on the terminal checkpoints of the matched arms.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
for arm in evid gated forced; do
  for s in 0 1 2; do
    out="diag_fixed/corr_m_${arm}_s${s}.json"
    [ -f "$out" ] && continue
    $PY tools/diag_corr_learned.py --config "configs/kitti/m_${arm}_s${s}.py" \
        --ckpt "work_dir/m_${arm}_s${s}/epoch_500.pth" --out "$out" --no-preload \
        > "diag_fixed/corr_m_${arm}_s${s}.log" 2>&1
    echo "$(date '+%m/%d %H:%M') diag_fixed ${arm} s${s} exit=$?" >> wacv_queue.log
  done
done
echo "$(date '+%m/%d %H:%M') diag_fixed sweep DONE" >> wacv_queue.log
