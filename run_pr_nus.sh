#!/usr/bin/env bash
# PR package req#1 -- nuScenes Car matrix, 4 arms x 5 paired seeds, GPU0.
# nuScenes val has 150 scenes, so SCENE (not tracklet) can be the bootstrap
# resampling unit -- KITTI's 2 Test scenes structurally cannot support this.
# All arms use CorrHeadMatched. Reporting checkpoint: fixed terminal epoch_40.pth.
# Host-RAM fix for the exit=137 kills lives in configs/nuscenes/car_cos_valid.py.
# Idempotent: finished arms skipped, interrupted arms auto-resume, one retry each.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
QLOG=wacv_queue.log
GPU=0
say() { echo "$(date '+%m/%d %H:%M') GPU$GPU $*" >> "$QLOG"; }

for s in 0 1 2 3 4; do for arm in base forced gated evid; do
  cfg="nus_${arm}_s${s}"
  [ -f "test_evals/$cfg.json" ] && { say "$cfg already evaluated, skip (pr-nus)"; continue; }

  for attempt in 1 2; do
    RESUME=""
    [ -f "work_dir/$cfg/last_checkpoint" ] && RESUME="--resume auto"
    say "launching $cfg attempt=$attempt ${RESUME:+(resume $(basename $(cat work_dir/$cfg/last_checkpoint)))} (pr-nus)"
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py "configs/nuscenes/$cfg.py" $RESUME >> "${cfg}.log" 2>&1
    rc=$?
    say "$cfg attempt=$attempt exit=$rc (pr-nus)"
    [ $rc -eq 0 ] && break
    [ $rc -eq 137 ] && say "$cfg exit=137 OOM-KILLED AGAIN -- RAM fix insufficient (pr-nus)"
    say "$cfg failed, retrying from last_checkpoint (pr-nus)"
  done

  if [ -f "work_dir/$cfg/epoch_40.pth" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/eval_tracklets.py "configs/nuscenes/$cfg.py" \
        "work_dir/$cfg/epoch_40.pth" --out "test_evals/$cfg.json" > "test_evals/$cfg.log" 2>&1
    say "$cfg eval rc=$? (pr-nus)"
  else
    say "$cfg NO epoch_40.pth -- eval skipped (pr-nus)"
  fi
done; done
say "pr-nus matrix DONE"
