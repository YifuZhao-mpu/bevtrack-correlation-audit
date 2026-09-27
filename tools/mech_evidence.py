"""Evidence-pathway counterfactual for the evidence-only head (inject='none').

The injection counterfactuals in tools/mech_closedloop.py cannot probe an
evidence-only arm: with inject='none' the injection branch is skipped, so
a0/shuffled/oracle all reduce to the normal rollout. But evid is exactly the
arm that beats the implicit baseline, and that comparison is NOT capacity
matched -- evid carries BEVCorrFuser's extra embed branch, which base lacks.
Capacity and correlation cannot be separated by adding an arm, because the
embed branch IS the correlation (zeroing its output at train time leaves it
without gradient, i.e. dead parameters, which controls for nothing).

The separable question is instead: does the head use the CONTENT of d_corr, or
only the fact that some extra branch feeds it? Three closed-loop rollouts of
the SAME trained evid model, differing only in what reaches reg():

  normal   : d_corr as estimated              (the model as trained)
  shuffled : d_corr drawn from another frame of this model's own normal pass
             -- right marginal distribution, wrong per-frame correspondence,
             and IN-distribution (this is the decisive condition)
  zero     : d_corr = 0 -- off-distribution, reported only as a reference bound

If shuffled falls back towards the implicit baseline, the gain comes from the
correlation content. If shuffled holds at normal, it comes from the branch's
capacity and the evidence claim does not survive.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/mech_evidence.py \
    --config configs/nuscenes/nus_evid_s0.py --ckpt work_dir/nus_evid_s0/epoch_40.pth \
    --tag nus_evid_s0 --out mech_ev/
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

MODES = ['normal', 'shuffled', 'zero']


def rollout(model, tracklet, mode, pool, rng, collect=None):
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
            head.evidence_mode = mode
            if mode == 'shuffled':
                head.evidence_override = torch.as_tensor(
                    pool[rng.integers(len(pool))], dtype=torch.float32)
            else:
                head.evidence_override = None
            coors = model.inference(data_dict)
            if collect is not None and head.last_disp is not None:
                collect.append(head.last_disp[0].cpu().numpy().tolist())
            cand = points_utils.getOffsetBB(
                ref_bb, [coors[0], coors[1], coors[2], coors[3]],
                degrees=True, use_z=True, limit_box=False)
            results_bbs.append(cand)
        ious.append(estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1]))
        dists.append(estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1]))
    head.evidence_mode = 'normal'; head.evidence_override = None
    return ious, dists, disps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True); ap.add_argument('--out', default='mech_ev/')
    ap.add_argument('--no-preload', action='store_true',
                    help='build the test set without the point-cloud preload cache. The '
                         'KITTI Car Test cache is 8.6 GB and will OOM-kill a concurrent '
                         'training run; per-frame disk reads cost RAM nothing and this '
                         'probe touches each frame a handful of times.')
    ap.add_argument('--limit', type=int, default=0, help='probe only the first N tracklets')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).cuda().eval()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck.get('state_dict', ck), strict=False)
    head = model.head
    assert hasattr(head, 'evidence_mode'), 'needs CorrHeadMatched with evidence control'
    assert head.inject == 'none', f"this probe is for the evidence-only arm, got inject={head.inject}"

    import copy
    dcfg = copy.deepcopy(cfg.test_dataloader['dataset'])
    if args.no_preload:
        dcfg['dataset']['preloading'] = False
    ds = DATASETS.build(dcfg)
    inner = getattr(ds, 'dataset', ds)
    nusc = getattr(inner, 'nusc', None)
    scenes = []
    for anno in getattr(inner, 'tracklet_anno_list', []):
        a = anno[0]
        s = a.get('scene', None)
        if s is None and nusc is not None and 'box_anno' in a:
            tok = a['box_anno'].get('sample_token')
            if tok:
                s = nusc.get('scene', nusc.get('sample', tok)['scene_token'])['name']
        scenes.append(str(s) if s is not None else '?')
    if any(s == '?' for s in scenes):
        sys.exit('FATAL: unresolved scene ids; refusing to run.')

    n_tr = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    out = {m: {'config': args.config, 'ckpt': args.ckpt, 'tracklets': []} for m in MODES}
    pool = []
    with torch.no_grad():
        # pass 1 builds the model's OWN d_corr pool that 'shuffled' draws from
        for i in range(n_tr):
            iou, dist, disp = rollout(model, ds[i], 'normal', None, rng, collect=pool)
            out['normal']['tracklets'].append(dict(
                index=i, n=len(iou), scene=scenes[i],
                iou=[round(float(x), 6) for x in iou],
                dist=[round(float(x), 6) for x in dist],
                disp=[round(float(x), 6) for x in disp]))
            if (i + 1) % 200 == 0:
                print(f'normal {i + 1}/{n_tr}', flush=True)
        pool = np.array(pool)
        print(f'd_corr pool: {pool.shape}', flush=True)
        for mode in ['shuffled', 'zero']:
            for i in range(n_tr):
                iou, dist, disp = rollout(model, ds[i], mode, pool, rng)
                out[mode]['tracklets'].append(dict(
                    index=i, n=len(iou), scene=scenes[i],
                    iou=[round(float(x), 6) for x in iou],
                    dist=[round(float(x), 6) for x in dist],
                    disp=[round(float(x), 6) for x in disp]))
                if (i + 1) % 200 == 0:
                    print(f'{mode} {i + 1}/{n_tr}', flush=True)
    for m in MODES:
        f = os.path.join(args.out, f'{args.tag}_{m}.json')
        json.dump(out[m], open(f, 'w')); print('saved', f)


if __name__ == '__main__':
    main()
