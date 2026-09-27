#!/usr/bin/env bash
# Rerun the per-frame / drift / recovery diagnostics on the matched arms at their
# terminal checkpoints, replacing reads of a pre-decode-fix correlation model at a
# Precision-selected checkpoint.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
for s in 0 1 2; do
  R="baseline:configs/kitti/m_base_s${s}.py:work_dir/m_base_s${s}/epoch_500.pth,corr:configs/kitti/m_gated_s${s}.py:work_dir/m_gated_s${s}/epoch_500.pth"
  for d in diag_head_xy diag_drift diag_recovery; do
    out="diag_fixed/${d}_s${s}.json"
    [ -f "$out" ] && continue
    $PY tools/${d}.py --runs "$R" --out "$out" --no-preload > "diag_fixed/${d}_s${s}.log" 2>&1
    echo "$(date '+%m/%d %H:%M') diag2 ${d} s${s} exit=$?" >> wacv_queue.log
  done
done
echo "$(date '+%m/%d %H:%M') diag2 sweep DONE" >> wacv_queue.log
