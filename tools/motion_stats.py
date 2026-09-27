"""Reality-check the SE(2)-Track thesis premise on nuScenes val: distribution of
inter-keyframe TARGET displacement ||Δcenter|| and |Δyaw|, per category.
Determines whether 'large-motion -> explicit (dx,dy,dθ) search' is justified, and
whether the rotation axis (dθ) matters or is ~0 in practice."""
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
val = set(splits.create_splits_scenes()['val'])
wrap = lambda d: (d + 180) % 360 - 180
pcts = [50, 75, 90, 95, 99]

print(f"{'cat':11s} {'pairs':>7s} | displacement ||Δc|| (m) p50/75/90/95/99 | |Δyaw| deg p50/75/90/95/99 | %mov>0.5m %yaw>5deg")
summary = {}
for cat in CATS:
    gens = set(tracking_to_general_class[cat])
    disp, dyaw = [], []
    for inst in nusc.instance:
        if nusc.get('category', inst['category_token'])['name'] not in gens:
            continue
        a0 = nusc.get('sample_annotation', inst['first_annotation_token'])
        sc = nusc.get('scene', nusc.get('sample', a0['sample_token'])['scene_token'])
        if sc['name'] not in val:
            continue
        tok = inst['first_annotation_token']
        pc, py = None, None
        while tok != '':
            ann = nusc.get('sample_annotation', tok)
            c = np.array(ann['translation'])
            q = Quaternion(ann['rotation']); y = q.degrees * q.axis[-1]
            if pc is not None:
                disp.append(np.linalg.norm(c[:2] - pc[:2]))
                dyaw.append(abs(wrap(y - py)))
            pc, py = c, y
            tok = ann['next']
    disp, dyaw = np.array(disp), np.array(dyaw)
    dp = [float(np.percentile(disp, p)) for p in pcts]
    yp = [float(np.percentile(dyaw, p)) for p in pcts]
    pmov = float((disp > 0.5).mean() * 100)
    pyaw = float((dyaw > 5).mean() * 100)
    summary[cat] = {'disp_pct': dp, 'yaw_pct': yp, 'pct_moving_0.5m': pmov, 'pct_yaw_gt5deg': pyaw, 'n_pairs': int(len(disp))}
    print(f"{cat:11s} {len(disp):7d} | {dp[0]:.2f}/{dp[1]:.2f}/{dp[2]:.2f}/{dp[3]:.2f}/{dp[4]:.2f} | "
          f"{yp[0]:.2f}/{yp[1]:.2f}/{yp[2]:.2f}/{yp[3]:.2f}/{yp[4]:.2f} | {pmov:.1f}% {pyaw:.1f}%")
json.dump(summary, open('/home/zyf/paper/refine-logs/motion_stats.json', 'w'), indent=2)
print("\nMOTION_STATS_DONE")
