"""Closed-loop matched-path counterfactual (round-3 gate 5, the causal proof).

The gated head's per-frame output is coors[:2] = r + g*inj. We run FOUR full
closed-loop rollouts of the same trained gated model, changing only the
injected term inj each frame:

  normal       : inj = d_corr      (the model as trained)
  a0           : inj = 0           (residual only; but r co-adapted with d_corr)
  shuffled_gt  : inj = a real inter-frame GT offset sampled from ANOTHER frame
                 (right marginal distribution of true motion, wrong value)
  shuffled_corr: inj = one of the model's OWN d_corr estimates sampled from
                 another frame of its normal rollout (right estimate
                 distribution incl. its biases, wrong value)
  oracle       : inj = d_gt in the CURRENT (drifted) reference frame

Two shuffles are reported because they control different things: shuffled_gt
asks "does injecting a plausible but wrong motion break it?", shuffled_corr
asks "is it the per-frame correspondence of the estimate that matters, rather
than its distribution?". (Earlier runs named the GT-pool control simply
"shuffled"; those files are the shuffled_gt condition.)

Each mode produces its own trajectory (predictions feed the next crop), so this
measures the closed-loop consequence of the injected estimate — where the
surrogate moving-target deficit lives (per-frame oracle-ref showed parity).

Per-tracklet iou/dist/disp are saved (eval_tracklets format) so
tools/paired_bootstrap.py gives tracklet-cluster CIs per mode.

Acceptance (surrogate, high apparent displacement): if `oracle` closes >=50%
of the normal-vs-baseline Success deficit and `a0`/`shuffled` do not, the
injected correlation estimate is the causal driver.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/mech_closedloop.py \
    --config configs/kitti/fix_gated_s0.py --ckpt work_dir/fix_gated_s0/epoch_500.pth \
    --split Test --tag surrogate_s0 --out mech_cl/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import datasets  # noqa: F401
import models    # noqa: F401
from datasets import points_utils
from datasets.metrics import estimateOverlap, estimateAccuracy

MODES = ['normal', 'a0', 'shuffled_gt', 'shuffled_corr', 'oracle']


def rollout(model, tracklet, mode, disp_pool, rng, collect_corr=None):
    """One closed-loop pass in the given injection mode. Returns iou/dist/disp lists.

    disp_pool: array the shuffle modes draw from (GT offsets or recorded d_corr).
    collect_corr: if a list, the model's own per-frame d_corr is appended to it
    (used by the normal pass to build the shuffled_corr pool).
    """
    head = model.head
    results_bbs = []
    ious, dists, disps = [], [], []
    for fid in range(len(tracklet)):
        this_bb = tracklet[fid]['3d_bbox']
        if fid == 0:
            results_bbs.append(this_bb)
            disps.append(0.0)
        else:
            prev_gt = tracklet[fid - 1]['3d_bbox']
            disps.append(float(np.linalg.norm(
                np.asarray(this_bb.center)[:2] - np.asarray(prev_gt.center)[:2])))
            data_dict, ref_bb = model.build_input_dict(tracklet, fid, results_bbs)
            # set injection for this frame
            head.inject_mode = mode
            if mode == 'oracle':
                d = ref_bb.rotation_matrix.T @ (np.asarray(this_bb.center) - np.asarray(ref_bb.center))
                head.inject_override = torch.tensor([float(d[0]), float(d[1])])
            elif mode in ('shuffled_gt', 'shuffled_corr'):
                head.inject_mode = 'shuffled'   # head only distinguishes override-vs-not
                head.inject_override = torch.as_tensor(disp_pool[rng.integers(len(disp_pool))],
                                                       dtype=torch.float32)
            else:
                head.inject_override = None
            coors = model.inference(data_dict)
            if collect_corr is not None and head.last_disp is not None:
                collect_corr.append(head.last_disp[0].cpu().numpy().tolist())
            cand = points_utils.getOffsetBB(
                ref_bb, [coors[0], coors[1], coors[2], coors[3]],
                degrees=True, use_z=True, limit_box=False)
            results_bbs.append(cand)
        ious.append(estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1]))
        dists.append(estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1]))
    head.inject_mode = 'normal'; head.inject_override = None
    return ious, dists, disps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='Test'); ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='mech_cl/')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'
    rng = np.random.default_rng(0)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(dev).eval()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)
    assert hasattr(model.head, 'inject_mode'), 'needs the gated head with inject control'

    dcfg = dict(cfg.test_dataloader['dataset'])
    dcfg['dataset'] = dict(dcfg['dataset']); dcfg['dataset']['split'] = args.split
    ds = DATASETS.build(dcfg)

    # displacement pool for 'shuffled' = per-frame GT offsets in ref frame (approx via consecutive GT)
    pool = []
    for i in range(len(ds)):
        tr = ds[i]
        for fid in range(1, len(tr)):
            a, b = tr[fid - 1]['3d_bbox'], tr[fid]['3d_bbox']
            d = a.rotation_matrix.T @ (np.asarray(b.center) - np.asarray(a.center))
            pool.append([float(d[0]), float(d[1])])
    pool = np.array(pool)

    # pass 1: normal rollouts, recording the model's own d_corr to build the
    # shuffled_corr pool (must exist before that mode can run).
    out = {m: {'tracklets': []} for m in MODES}
    corr_pool = []
    with torch.no_grad():
        for i in range(len(ds)):
            iou, dist, disp = rollout(model, ds[i], 'normal', pool, rng, collect_corr=corr_pool)
            out['normal']['tracklets'].append(dict(
                index=i, n=len(iou),
                iou=[round(float(x), 6) for x in iou],
                dist=[round(float(x), 6) for x in dist],
                disp=[round(float(x), 6) for x in disp]))
    corr_pool = np.array(corr_pool) if corr_pool else pool
    print(f'normal pass done; d_corr pool {corr_pool.shape}')

    with torch.no_grad():
        for i in range(len(ds)):
            tr = ds[i]
            for m in MODES:
                if m == 'normal':
                    continue
                src = corr_pool if m == 'shuffled_corr' else pool
                iou, dist, disp = rollout(model, tr, m, src, rng)
                out[m]['tracklets'].append(dict(
                    index=i, n=len(iou),
                    iou=[round(float(x), 6) for x in iou],
                    dist=[round(float(x), 6) for x in dist],
                    disp=[round(float(x), 6) for x in disp]))
            if (i + 1) % 30 == 0:
                print(f'{i + 1}/{len(ds)} tracklets')
    for m in MODES:
        json.dump(out[m], open(f'{args.out}/{args.tag}_{m}.json', 'w'))
    print('saved', args.tag, MODES)


if __name__ == '__main__':
    main()
