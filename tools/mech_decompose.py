"""Causal decomposition of the injected-branch effect (review round 4, req #6).

The earlier closed-loop counterfactual (tools/mech_closedloop.py) showed a large
oracle-vs-normal gap but could not say WHY, because feeding a counterfactual
forward changes every later crop. Round 4 also showed ~6 of the ~10 oracle points
appear on the FIRST predicted transition, i.e. most of the benefit is immediate,
not compounded. This script separates the components properly.

Four quantities, all on the SAME natural trajectory states (no counterfactual is
ever fed forward except in the explicit rollout arms):

  fixed-state  : replay each natural state; swap only the injected term; measure
                 the one-step error. Isolates the instantaneous estimator effect.
  dose         : d_lambda = d_corr + lambda * (d_GT - d_corr), lambda in
                 {0, .25, .5, .75, 1} at fixed states. A monotone response is
                 required before any "estimate quality drives tracking" claim.
  reset        : full rollout with the NATURAL learned estimate, but the next
                 reference is reset to the previous GT box. Removes accumulated
                 crop drift WITHOUT granting GT displacement to the output, so it
                 isolates feedback amplification.
  shuffle_corr : permute the model's own d_corr WITHIN bins of (apparent motion,
                 pre-step drift, |d_corr|, confidence), so the marginal
                 distribution is preserved and only the per-frame correspondence
                 is destroyed.

Per-frame logging (npz): d_corr, d_gt, r, g, conf, pre-step drift, post-step
error, scene, track id — so the transition-chain analysis (does current
correlation error predict next-state drift?) can be done offline.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/mech_decompose.py \
      --config configs/kitti/m_gated_s0.py --ckpt work_dir/m_gated_s0/epoch_500.pth \
      --split Test --tag kitti_m_s0 --out mech_dec/
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

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def gt_offset(ref_box, this_box):
    d = ref_box.rotation_matrix.T @ (np.asarray(this_box.center) - np.asarray(ref_box.center))
    return np.array([float(d[0]), float(d[1])])


def natural_pass(model, tracklet, scene):
    """Natural rollout; records per-frame state and the model's own quantities."""
    head = model.head
    head.inject_mode, head.inject_override = 'normal', None
    boxes, rows = [], []
    for fid in range(len(tracklet)):
        this_bb = tracklet[fid]['3d_bbox']
        if fid == 0:
            boxes.append(this_bb)
            continue
        data_dict, ref_bb = model.build_input_dict(tracklet, fid, boxes)
        coors = model.inference(data_dict)
        cand = points_utils.getOffsetBB(ref_bb, [coors[0], coors[1], coors[2], coors[3]],
                                        degrees=True, use_z=True, limit_box=False)
        prev_gt = tracklet[fid - 1]['3d_bbox']
        d_gt = gt_offset(ref_bb, this_bb)
        d_corr = head.last_disp[0].cpu().numpy() if head.last_disp is not None else np.zeros(2)
        g = float(head.last_gate[0, 0]) if head.last_gate is not None else 1.0
        rows.append(dict(
            fid=fid, scene=scene,
            d_corr=d_corr.tolist(), d_gt=d_gt.tolist(), g=g,
            # pre-step drift: how far the reference the tracker used is from the true previous box
            drift=float(np.linalg.norm(np.asarray(ref_bb.center)[:2] - np.asarray(prev_gt.center)[:2])),
            apparent=float(np.linalg.norm(np.asarray(this_bb.center)[:2] - np.asarray(prev_gt.center)[:2])),
            err=float(np.linalg.norm(np.array([coors[0], coors[1]]) - d_gt)),
            corr_err=float(np.linalg.norm(d_corr - d_gt)),
        ))
        boxes.append(cand)
    return boxes, rows


def fixed_state_errors(model, tracklet, boxes, rows):
    """Replay the natural states; swap only the injected term. One-step xy error."""
    head = model.head
    out = {f'lam{l}': [] for l in LAMBDAS}
    out['a0'] = []
    for k, row in enumerate(rows):
        fid = row['fid']
        data_dict, ref_bb = model.build_input_dict(tracklet, fid, boxes[:fid])
        d_gt = np.array(row['d_gt']); d_corr = np.array(row['d_corr'])
        for lam in LAMBDAS:
            head.inject_mode = 'oracle'
            head.inject_override = torch.as_tensor(d_corr + lam * (d_gt - d_corr), dtype=torch.float32)
            c = model.inference(data_dict)
            out[f'lam{lam}'].append(float(np.linalg.norm(np.array([c[0], c[1]]) - d_gt)))
        head.inject_mode, head.inject_override = 'a0', None
        c = model.inference(data_dict)
        out['a0'].append(float(np.linalg.norm(np.array([c[0], c[1]]) - d_gt)))
    head.inject_mode, head.inject_override = 'normal', None
    return out


def reset_rollout(model, tracklet):
    """Natural estimate, but the next reference is reset to the previous GT box."""
    head = model.head
    head.inject_mode, head.inject_override = 'normal', None
    ious, dists, disps = [], [], []
    boxes = []
    for fid in range(len(tracklet)):
        this_bb = tracklet[fid]['3d_bbox']
        if fid == 0:
            boxes.append(this_bb); disps.append(0.0)
        else:
            prev_gt = tracklet[fid - 1]['3d_bbox']
            disps.append(float(np.linalg.norm(np.asarray(this_bb.center)[:2] - np.asarray(prev_gt.center)[:2])))
            # reference = previous GT (drift removed), prediction still from the learned estimate
            data_dict, _ = model.build_input_dict(tracklet, fid, boxes[:fid - 1] + [prev_gt])
            coors = model.inference(data_dict)
            cand = points_utils.getOffsetBB(prev_gt, [coors[0], coors[1], coors[2], coors[3]],
                                            degrees=True, use_z=True, limit_box=False)
            boxes.append(cand)
        ious.append(estimateOverlap(this_bb, boxes[-1], dim=3, up_axis=[0, 0, 1]))
        dists.append(estimateAccuracy(this_bb, boxes[-1], dim=3, up_axis=[0, 0, 1]))
    return ious, dists, disps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='Test'); ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='mech_dec/'); ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).cuda().eval()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)
    assert hasattr(model.head, 'inject_mode'), 'needs a head with injection control'

    dcfg = dict(cfg.test_dataloader['dataset'])
    dcfg['dataset'] = dict(dcfg['dataset']); dcfg['dataset']['split'] = args.split
    ds = DATASETS.build(dcfg)
    inner = getattr(ds, 'dataset', ds)
    scenes = [str(a[0].get('scene', a[0].get('scene_token', '?')))
              for a in getattr(inner, 'tracklet_anno_list', [])]

    n_tr = len(ds) if not args.limit else min(args.limit, len(ds))
    all_rows, fixed, reset_tr = [], {f'lam{l}': [] for l in LAMBDAS}, []
    fixed['a0'] = []
    with torch.no_grad():
        for i in range(n_tr):
            tr = ds[i]
            sc = scenes[i] if i < len(scenes) else '?'
            boxes, rows = natural_pass(model, tr, sc)
            if not rows:
                continue
            fs = fixed_state_errors(model, tr, boxes, rows)
            for k, v in fs.items():
                fixed[k].extend(v)
            for r, e in zip(rows, fs[f'lam0.0']):
                r['fixed_lam0'] = e
            all_rows.extend(rows)
            iou, dist, disp = reset_rollout(model, tr)
            reset_tr.append(dict(index=i, n=len(iou), scene=sc,
                                 iou=[round(float(x), 6) for x in iou],
                                 dist=[round(float(x), 6) for x in dist],
                                 disp=[round(float(x), 6) for x in disp]))
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{n_tr}')

    summ = {'tag': args.tag, 'split': args.split, 'n_frames': len(all_rows),
            'fixed_state_mean_err': {k: float(np.mean(v)) for k, v in fixed.items()},
            'dose_monotone': None}
    lam_means = [summ['fixed_state_mean_err'][f'lam{l}'] for l in LAMBDAS]
    summ['dose_monotone'] = bool(all(lam_means[i] >= lam_means[i + 1] for i in range(len(lam_means) - 1)))
    summ['gate_mean'] = float(np.mean([r['g'] for r in all_rows]))
    summ['corr_err_mean'] = float(np.mean([r['corr_err'] for r in all_rows]))
    json.dump(summ, open(f'{args.out}/{args.tag}_summary.json', 'w'), indent=2)
    json.dump({'tracklets': reset_tr}, open(f'{args.out}/{args.tag}_reset.json', 'w'))
    np.savez(f'{args.out}/{args.tag}_frames.npz',
             **{k: np.array([r[k] for r in all_rows]) for k in
                ['fid', 'g', 'drift', 'apparent', 'err', 'corr_err', 'fixed_lam0']},
             d_corr=np.array([r['d_corr'] for r in all_rows]),
             d_gt=np.array([r['d_gt'] for r in all_rows]),
             scene=np.array([r['scene'] for r in all_rows]))
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
