#!/usr/bin/env bash
# Recompute every printed interval at a high, fixed number of draws so the
# endpoints are stable to the precision the paper prints them at. The earlier
# values were produced at 600-4000 draws and an audit found 13 endpoints
# outside the Monte-Carlo range of independent reruns.
#
# The estimator is seeded (--seed 0) and the draw counts are fixed here, so a
# rerun reproduces the endpoints exactly rather than to within Monte-Carlo error.
set -eu

PY=${PY:-python}
OUT=${OUT:-intervals_final.txt}
cd "$(dirname "$0")"
: > "$OUT"

run() { echo "### $1" >> "$OUT"; shift; "$PY" tools/seed_scene_ci.py "$@" 2>&1 \
        | grep -E "^  (all|static|moving)|only|seed\+|t-int" >> "$OUT"; }

# Three KITTI seeds per arm. NOTE: this helper took `local a=$1 b=$2` until
# 2026-09-25 while every call site passes one argument; under `set -u` the
# unset $2 aborted the subshell, so `$(K base)` expanded to the empty string
# and the three KITTI runs wrote a header with no rows beneath it.
K() { local a=$1; echo "test_evals/m_${a}_s0.json,test_evals/m_${a}_s1.json,test_evals/m_${a}_s2.json"; }
for arm in forced gated evid; do
  run "KITTI $arm (40000 draws)" "$(K base)" "$(K "$arm")" --unit tracklet --n-boot 40000
done

# Five nuScenes seeds per arm, resampled over scene clusters.
N() { local a=$1; echo "test_evals/nus_${a}_s0.json,test_evals/nus_${a}_s1.json,test_evals/nus_${a}_s2.json,test_evals/nus_${a}_s3.json,test_evals/nus_${a}_s4.json"; }
for arm in forced gated evid; do
  run "nuScenes $arm (20000 draws)" "$(N base)" "$(N "$arm")" --n-boot 20000
done

# Appendix four-category sweep on the restored encoder.
for cat in "" ped_ van_ cyc_; do
  for arm in forced gated evid; do
    n=3; [ "$cat" = "ped_" ] && [ "$arm" = "evid" ] && n=2
    a=""; b=""
    for k in $(seq 0 $((n-1))); do
      a="$a,test_evals/fix_${cat}base_s${k}.json"
      b="$b,test_evals/fix_${cat}${arm}_s${k}.json"
    done
    run "APP ${cat:-car_}$arm (40000 draws)" "${a:1}" "${b:1}" --unit tracklet --n-boot 40000
  done
done

run "OFFICIAL gated (40000 draws)" \
  "test_evals/official_base_s0.json,test_evals/official_base_s1.json,test_evals/official_base_s2.json" \
  "test_evals/official_gated_s0.json,test_evals/official_gated_s1.json,test_evals/official_gated_s2.json" \
  --unit tracklet --n-boot 40000

echo "wrote $OUT"
