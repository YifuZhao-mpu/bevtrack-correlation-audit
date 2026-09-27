"""Frozen per-tracklet evaluation (review 2026-07-18 protocol).

Runs one closed-loop pass over every tracklet of the config's test split and
saves PER-TRACKLET per-frame arrays (iou, center distance, gt displacement),
so any aggregate (pooled AUC, strata, tracklet-cluster bootstrap) can be
recomputed offline without re-running inference.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/eval_tracklets.py <config> <ckpt> --out <json>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import datasets  # noqa: F401
import models    # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('ckpt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-preload', action='store_true',
                    help='build the test set without the point-cloud preload cache '
                         '(KITTI Car Test is 8.6 GB and will OOM-kill a concurrent '
                         'training run). Data content is identical; only the caching '
                         'differs. Default is unchanged, so queued evals are unaffected.')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).cuda().eval()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('missing:', missing, 'unexpected:', unexpected)

    import copy
    _dcfg = copy.deepcopy(cfg.test_dataloader['dataset'])
    if args.no_preload:
        _dcfg['dataset']['preloading'] = False
    ds = DATASETS.build(_dcfg)
    # scene id per tracklet: the true clustering unit for inference (KITTI Test has
    # only 2 scenes and whole categories live in one of them; nuScenes val has 150).
    # KITTI annos are pandas rows carrying a 'scene' column (datasets/kitti.py:123).
    # nuScenes annos are {'sample_data_lidar', 'box_anno'} ONLY (datasets/nuscenes.py
    # _build_tracklet_anno), so the scene must be resolved through the nusc index --
    # otherwise every tracklet silently collapses to '?' and scene-level resampling,
    # the entire reason for running the nuScenes matrix, becomes impossible while the
    # JSON still looks valid. Verified 2026-09-12.
    inner = getattr(ds, 'dataset', ds)
    nusc = getattr(inner, 'nusc', None)
    scenes = []
    for anno in getattr(inner, 'tracklet_anno_list', []):
        a = anno[0]
        s = a.get('scene', None)
        if s is None:
            s = a.get('scene_token', None) or a.get('scene_name', None)
        if s is None and nusc is not None and 'box_anno' in a:
            tok = a['box_anno'].get('sample_token')
            if tok:
                s = nusc.get('scene', nusc.get('sample', tok)['scene_token'])['name']
        scenes.append(str(s) if s is not None else '?')

    n_unres = sum(1 for s in scenes if s == '?')
    if not scenes or n_unres:
        print(f'FATAL: {n_unres}/{len(scenes)} tracklets have no resolvable scene id; '
              'scene-level resampling would be impossible and the output JSON would '
              'still look valid. Refusing to run inference.', file=sys.stderr)
        sys.exit(2)
    print(f'scene ids resolved: {len(scenes)} tracklets across '
          f'{len(set(scenes))} distinct scenes')
    out = {'config': args.config, 'ckpt': args.ckpt, 'tracklets': []}
    with torch.no_grad():
        for i in range(len(ds)):
            tracklet = ds[i]
            ious, dists, disps = model.predict(tracklet)
            out['tracklets'].append(dict(
                index=i, n=len(ious), scene=(scenes[i] if i < len(scenes) else '?'),
                iou=[round(float(x), 6) for x in ious],
                dist=[round(float(x), 6) for x in dists],
                disp=[round(float(x), 6) for x in disps]))
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(ds)} tracklets')
    json.dump(out, open(args.out, 'w'))
    print('saved', args.out)


if __name__ == '__main__':
    main()
