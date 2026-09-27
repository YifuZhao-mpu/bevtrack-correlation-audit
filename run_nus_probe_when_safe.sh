#!/usr/bin/env bash
# Run the nuScenes evidence-pathway probe only in a window where it cannot
# OOM-kill a training run: no nuScenes training resident (its 9.1 GB preload
# expands to ~26 GB) and >=18 GB of host RAM actually free. The probe itself
# needs ~5-6 GB for the nusc index. --limit keeps it inside a short window.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
for i in $(seq 1 2880); do            # up to 48h of waiting, 60s apart
  avail=$(free -g | sed -n '2p' | awk '{print $NF}')
  nus_running=$(pgrep -fc "train.py configs/nuscenes" || true)
  if [ "${nus_running:-0}" -eq 0 ] && [ "$avail" -ge 18 ]; then
    echo "$(date '+%m/%d %H:%M') window open (avail=${avail}GB, no nus training) - launching probe" >> wacv_queue.log
    $PY tools/mech_evidence.py --config configs/nuscenes/nus_evid_s0.py \
        --ckpt work_dir/nus_evid_s0/epoch_40.pth --tag nus_evid_s0 \
        --out mech_ev/ --limit 300 > mech_ev_nus_s0.log 2>&1
    echo "$(date '+%m/%d %H:%M') nus evidence probe exit=$? " >> wacv_queue.log
    exit 0
  fi
  sleep 60
done
echo "$(date '+%m/%d %H:%M') nus evidence probe: no safe window in 48h" >> wacv_queue.log
exit 1
