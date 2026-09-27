#!/bin/bash
source /home/zyf/anaconda3/etc/profile.d/conda.sh
cd /home/zyf/paper/BEVTrack
echo "[wait] waiting for se2track env to be importable..."
ready=0
for i in $(seq 1 120); do
  if conda run -n se2track python -c "import torch,mmengine,nuscenes,pyquaternion,shapely,scipy" 2>/dev/null; then
    echo "[wait] ENV READY after ~$((i*20))s"; ready=1; break
  fi
  sleep 20
done
[ "$ready" = 1 ] || { echo "[wait] ENV NOT READY (timeout) — check env_setup_311.log"; exit 1; }

run() { echo; echo "######## $1 ########"; CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n se2track python "$2" 2>&1; }
run "smoke_encoder (sm_120 fwd/bwd)" tools/smoke_encoder.py
run "m0_geometry (convention round-trip)" tools/m0_geometry_test.py
run "m0_metric (Success/Precision)" tools/m0_metric_test.py
run "m0_data (real nuScenes pipeline)" tools/m0_data_test.py
echo; echo "M0_SEQUENCE_DONE"
