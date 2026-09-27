"""Freeze per-category stratification terciles on nuScenes val (for M2 pilot gate).
large-yaw stratum = |GT yaw-change between consecutive keyframes| above per-category UPPER tercile.
low-point stratum = target num_lidar_pts below per-category LOWER tercile.
Uses metadata only (num_lidar_pts + box yaw from quaternion) -> fast, no point-cloud loads.
Writes strata.json. LOCK this before any training/eval.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.splits as splits
from datasets.nuscenes import tracking_to_general_class

DATA = '/home/zyf/data/nuscenes'
CATS = ['car', 'pedestrian', 'truck', 'trailer', 'bus']
nusc = NuScenes(version='v1.0-trainval', dataroot=DATA, verbose=False)
val_scenes = set(splits.create_splits_scenes()['val'])


def yaw_of(ann):
    q = Quaternion(ann['rotation'])
    return q.degrees * q.axis[-1]


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


out = {}
for cat in CATS:
    gens = set(tracking_to_general_class[cat])
    dyaws, npts = [], []
    for inst in nusc.instance:
        cname = nusc.get('category', inst['category_token'])['name']
        if cname not in gens:
            continue
        # first annotation's scene must be in val
        a0 = nusc.get('sample_annotation', inst['first_annotation_token'])
        scene = nusc.get('scene', nusc.get('sample', a0['sample_token'])['scene_token'])
        if scene['name'] not in val_scenes:
            continue
        tok = inst['first_annotation_token']
        prev_yaw = None
        while tok != '':
            ann = nusc.get('sample_annotation', tok)
            npts.append(ann['num_lidar_pts'])
            y = yaw_of(ann)
            if prev_yaw is not None:
                dyaws.append(abs(wrap180(y - prev_yaw)))
            prev_yaw = y
            tok = ann['next']
    dyaws, npts = np.array(dyaws), np.array(npts)
    out[cat] = {
        'n_frames': int(len(npts)), 'n_pairs': int(len(dyaws)),
        'yaw_upper_tercile_deg': float(np.percentile(dyaws, 66.67)) if len(dyaws) else None,
        'point_lower_tercile': float(np.percentile(npts, 33.33)) if len(npts) else None,
        'yaw_median_deg': float(np.median(dyaws)) if len(dyaws) else None,
        'point_median': float(np.median(npts)) if len(npts) else None,
    }
    print(f"{cat:11s} frames={out[cat]['n_frames']:6d} pairs={out[cat]['n_pairs']:6d} "
          f"yaw_upper={out[cat]['yaw_upper_tercile_deg']:.2f}deg point_lower={out[cat]['point_lower_tercile']:.0f} "
          f"(yaw_med={out[cat]['yaw_median_deg']:.2f} pt_med={out[cat]['point_median']:.0f})")

dst = '/home/zyf/paper/refine-logs/strata.json'
json.dump(out, open(dst, 'w'), indent=2)
print(f"\nwrote {dst}")
print("STRATA_FROZEN")
