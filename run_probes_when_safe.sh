#!/usr/bin/env bash
# Run the evidence-pathway probe on every evid checkpoint that does not have one
# yet, each inside a window where it cannot OOM-kill a training run.
# nuScenes needs ~5-6 GB for the nusc index and must not overlap a nuScenes
# training (whose 9.1 GB preload expands to ~26 GB). KITTI uses --no-preload and
# is cheap, so it only needs modest headroom.
set -u
PY=/home/zyf/anaconda3/envs/se2track/bin/python
cd /home/zyf/paper/BEVTrack
QLOG=wacv_queue.log
say() { echo "$(date '+%m/%d %H:%M') $*" >> "$QLOG"; }

pending() {   # echo "<dataset> <cfgdir> <cfg> <ckptname> <minram>" for each missing probe
  for s in 0 1 2 3 4; do
    [ -f "work_dir/nus_evid_s$s/epoch_40.pth" ] && [ ! -f "mech_ev/nus_evid_s${s}_shuffled.json" ] \
      && echo "nus configs/nuscenes nus_evid_s$s epoch_40.pth 18"
  done
  for s in 0 1 2; do
    [ -f "work_dir/m_evid_s$s/epoch_500.pth" ] && [ ! -f "mech_ev/kitti_evid_s${s}_shuffled.json" ] \
      && echo "kitti configs/kitti m_evid_s$s epoch_500.pth 8"
  done
}

for i in $(seq 1 4320); do            # up to 72h
  job=$(pending | head -1)
  if [ -z "$job" ]; then sleep 60; continue; fi
  set -- $job; ds=$1; cdir=$2; cfg=$3; ck=$4; minram=$5
  avail=$(free -g | sed -n '2p' | awk '{print $NF}')
  nus_running=$(pgrep -fc "train.py configs/nuscenes" || true)
  ok=0
  if [ "$ds" = "nus" ]; then
    [ "${nus_running:-0}" -eq 0 ] && [ "$avail" -ge "$minram" ] && ok=1
  else
    [ "$avail" -ge "$minram" ] && ok=1
  fi
  if [ "$ok" -eq 1 ]; then
    tag=$([ "$ds" = "nus" ] && echo "$cfg" || echo "kitti_evid_s${cfg##*_s}")
    extra=$([ "$ds" = "kitti" ] && echo "--no-preload" || echo "--limit 300")
    say "probe window open (avail=${avail}GB) - $cfg"
    $PY tools/mech_evidence.py --config "$cdir/$cfg.py" --ckpt "work_dir/$cfg/$ck" \
        --tag "$tag" --out mech_ev/ $extra > "mech_ev_${tag}.log" 2>&1
    say "probe $tag exit=$?"
  fi
  sleep 60
done
say "probe runner: finished sweep"
