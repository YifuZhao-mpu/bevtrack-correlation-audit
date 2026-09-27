#!/usr/bin/env bash
# PR package req#3 -- KITTI Car matched-head matrix, 4 arms x 3 seeds, GPU1.
# The legacy CorrHead was capacity/input-unmatched (see configs/kitti/m_forced_s0.py);
# CorrHeadMatched holds residual inputs (feat+disp+conf) and capacity fixed so that
# ONLY the injection rule differs. Reporting checkpoint: fixed terminal epoch_500.pth.
# Idempotent: finished arms are skipped, interrupted arms auto-resume from
# last_checkpoint, and one transient-CUDA-fault retry is allowed per arm.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
QLOG=wacv_queue.log
GPU=1
say() { echo "$(date '+%m/%d %H:%M') GPU$GPU $*" >> "$QLOG"; }

for s in 0 1 2; do for arm in base forced gated evid; do
  cfg="m_${arm}_s${s}"
  [ -f "test_evals/$cfg.json" ] && { say "$cfg already evaluated, skip (pr-matched)"; continue; }

  for attempt in 1 2; do
    RESUME=""
    [ -f "work_dir/$cfg/last_checkpoint" ] && RESUME="--resume auto"
    say "launching $cfg attempt=$attempt ${RESUME:+(resume $(basename $(cat work_dir/$cfg/last_checkpoint)))} (pr-matched)"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/kitti/$cfg.py" $RESUME >> "${cfg}.log" 2>&1
    rc=$?
    say "$cfg attempt=$attempt exit=$rc (pr-matched)"
    [ $rc -eq 0 ] && break
    say "$cfg failed, retrying from last_checkpoint (pr-matched)"
  done

  if [ -f "work_dir/$cfg/epoch_500.pth" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/eval_tracklets.py "configs/kitti/$cfg.py" \
        "work_dir/$cfg/epoch_500.pth" --out "test_evals/$cfg.json" > "test_evals/$cfg.log" 2>&1
    say "$cfg eval rc=$? (pr-matched)"
  else
    say "$cfg NO epoch_500.pth -- eval skipped (pr-matched)"
  fi
done; done
say "pr-matched matrix DONE"
