"""M2 diagnostic part 4: one-step recovery from drift (the closed-loop amplifier).

From a reference drifted by delta in {0.5, 1.0} m, one prediction step yields
residual r = |pred - gt|. 'Recovered' = r <= 0.5 m (back inside the benign
regime where both heads are accurate). Reports recovery rates at thresholds
{0.3, 0.5} and dumps raw per-frame residuals (diag_recovery.npz) so any
threshold can be recomputed. Writes diag_recovery.json.
Run: CUDA_VISIBLE_DEVICES=1 python tools/diag_recovery.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
import datasets  # noqa
import models    # noqa
from datasets import points_utils

RNG = 4.8
OFFSETS = [0.5, 1.0]
THRESH = [0.3, 0.5]
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
    rng = np.random.RandomState(0)
    cfg0 = Config.fromfile(RUNS[0][1])
    if args.no_preload:
        cfg0.test_dataloader['dataset']['dataset']['preloading'] = False
    ds = DATASETS.build(cfg0.test_dataloader['dataset'])
    samples = getattr(ds, 'training_samples', None) or getattr(ds, 'tracklets', None)
    if samples is None:
        samples = [ds[i] for i in range(len(ds))]

    frame_dirs = {}
    for ti, tr in enumerate(samples):
        for fid in range(1, len(tr)):
            a = rng.uniform(0, 2 * np.pi)
            frame_dirs[(ti, fid)] = np.array([np.cos(a), np.sin(a)])

    out, raw = {}, {}
    for name, cfg_path, ckpt in RUNS:
        cfg = Config.fromfile(cfg_path)
        model = MODELS.build(cfg.model).to(dev).eval()
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'])
        errs = {s: {o: [] for o in OFFSETS} for s in ('static', 'inband')}
        with torch.no_grad():
            for ti, tr in enumerate(samples):
                for fid in range(1, len(tr)):
                    ref0 = tr[fid - 1]['3d_bbox']; cur = tr[fid]['3d_bbox']
                    d0 = ref0.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref0.center))
                    mag0 = float(np.linalg.norm(d0[:2]))
                    if mag0 > RNG:
                        continue
                    stratum = 'static' if mag0 < 0.5 else 'inband'
                    u = frame_dirs[(ti, fid)]
                    for off in OFFSETS:
                        ref = points_utils.getOffsetBB(
                            ref0, [off * u[0], off * u[1], 0.0, 0.0],
                            degrees=True, use_z=True, limit_box=False)
                        d = ref.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref.center))
                        gt = np.array([float(d[0]), float(d[1])])
                        prev_pts = points_utils.crop_pc_in_range(tr[fid-1]['pc'], ref, cfg.point_cloud_range).points.T
                        this_pts = points_utils.crop_pc_in_range(tr[fid]['pc'], ref, cfg.point_cloud_range).points.T
                        if prev_pts.shape[0] < 1 or this_pts.shape[0] < 1:
                            continue
                        inputs = {'prev_points': [torch.as_tensor(prev_pts, dtype=torch.float32, device=dev)],
                                  'this_points': [torch.as_tensor(this_pts, dtype=torch.float32, device=dev)]}
                        coors = model.inference(inputs)
                        errs[stratum][off].append(float(np.linalg.norm(coors[:2] - gt)))
        res = {}
        for s, so in errs.items():
            res[s] = {}
            for o, v in so.items():
                a = np.array(v)
                raw[f'{name}_{s}_{o}'] = a
                res[s][str(o)] = dict(
                    n=len(a), mean_err=float(a.mean()),
                    **{f'recov_le_{t}': float(100 * (a <= t).mean()) for t in THRESH})
        out[name] = res
        print(name, json.dumps(res, indent=2))

    json.dump(out, open(args.out, 'w'), indent=2)
    print('saved', args.out)
    np.savez(args.out.replace('.json', '.npz'), **raw)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
