"""M2 diagnostic part 2: per-frame FINAL (dx,dy) error of full heads under oracle refs.

Compares baseline SimpleHead vs corr v3 head end-to-end regression quality per stratum,
isolating per-frame accuracy from tracking-drift dynamics. Sets the quantitative bar any
corr v4 must beat (esp. inband). Writes diag_head_xy.json.
Run: CUDA_VISIBLE_DEVICES=0 python tools/diag_head_xy.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
import datasets
import models
from datasets import points_utils

RNG = 4.8
# RUNS is supplied on the command line. The superseded default probed
# ckpts/none_best_ep122.pth and work_dir/car_corr2/best_precision_epoch_450.pth
# -- a pre-decode-fix correlation model and two Precision/Success-selected
# checkpoints, i.e. neither an arm this paper reports nor a selection-free read.
RUNS = []


def main():
    global RUNS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True,
                    help="comma-separated name:config:ckpt triples")
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-preload', action='store_true')
    args = ap.parse_args()
    RUNS = [tuple(r.split(':')) for r in args.runs.split(',')]
    dev = 'cuda'
    cfg0 = Config.fromfile(RUNS[0][1])
    if args.no_preload:
        cfg0.test_dataloader['dataset']['dataset']['preloading'] = False
    ds = DATASETS.build(cfg0.test_dataloader['dataset'])
    samples = getattr(ds, 'training_samples', None) or getattr(ds, 'tracklets', None)
    if samples is None:
        samples = [ds[i] for i in range(len(ds))]

    out = {}
    for name, cfg_path, ckpt in RUNS:
        cfg = Config.fromfile(cfg_path)
        model = MODELS.build(cfg.model).to(dev).eval()
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'])
        errs = {'static': [], 'inband': []}
        with torch.no_grad():
            for tr in samples:
                for fid in range(1, len(tr)):
                    ref = tr[fid - 1]['3d_bbox']; cur = tr[fid]['3d_bbox']
                    d = ref.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref.center))
                    gt = np.array([float(d[0]), float(d[1])])
                    mag = float(np.linalg.norm(gt))
                    if mag > RNG:
                        continue
                    stratum = 'static' if mag < 0.5 else 'inband'
                    prev_pts = points_utils.crop_pc_in_range(tr[fid-1]['pc'], ref, cfg.point_cloud_range).points.T
                    this_pts = points_utils.crop_pc_in_range(tr[fid]['pc'], ref, cfg.point_cloud_range).points.T
                    if prev_pts.shape[0] < 1 or this_pts.shape[0] < 1:
                        continue
                    inputs = {'prev_points': [torch.as_tensor(prev_pts, dtype=torch.float32, device=dev)],
                              'this_points': [torch.as_tensor(this_pts, dtype=torch.float32, device=dev)]}
                    coors = model.inference(inputs)
                    errs[stratum].append((float(np.linalg.norm(coors[:2] - gt)), mag))
        res = {}
        for s, rows in errs.items():
            e = np.array([r[0] for r in rows]); m = np.array([r[1] for r in rows])
            res[s] = dict(n=len(rows), mean_err=float(e.mean()), median_err=float(np.median(e)),
                          p90_err=float(np.quantile(e, 0.9)),
                          worse_than_zero_pct=float(100 * np.mean(e > m)))
        out[name] = res
        print(name, json.dumps(res, indent=2))

    json.dump(out, open(args.out, 'w'), indent=2)
    print('saved', args.out)


if __name__ == '__main__':
    main()
