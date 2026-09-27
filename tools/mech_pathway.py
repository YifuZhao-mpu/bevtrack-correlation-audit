"""Matched-path counterfactual (round-3 gate 5, oracle-reference one-step).

On a trained GATED model, the head forms coors[:2] = r + g*d_corr, where the
residual r ALSO sees (d_corr, conf) as features. We hold r and g fixed per
frame and swap ONLY the injected additive term d_corr:

  a1        : r + g*d_corr        (current model)
  a0        : r                   (drop injected path; evidence still in r)
  shuffled  : r + g*d_shuf        (d_corr permuted across same-stratum frames)
  oracle    : r + g*d_gt          (inject the GT offset for this reference)

Evaluated at reference drift delta in {0, 0.5, 1.0} m (oracle GT ref perturbed
by a fixed random direction per frame) — brackets the closed-loop drift regime
without a full rollout. For reference we also record the implicit BASELINE
model's error and the a0-vs-a1 error, so deficit closure
(err_a1 - err_a0)/(err_a1 - err_base) is computable offline per stratum.

Also dumps, per frame: gate g, corr error |d_corr - d_gt|, conf, injected-norm
|g*d_corr|, pre-step drift delta, apparent |d_gt|. Raw arrays -> npz so any
CI / threshold is recomputed offline (tools/paired_bootstrap-style).

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/mech_pathway.py \
      --gated-config configs/kitti/fix_gated_s0.py --gated-ckpt work_dir/fix_gated_s0/epoch_500.pth \
      --base-config  configs/kitti/fix_base_s0.py  --base-ckpt  work_dir/fix_base_s0/epoch_500.pth \
      --split Valid --tag surrogate_s0 --out mech/
Run on Valid FIRST (freeze), then Test once.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import datasets  # noqa: F401
import models    # noqa: F401
from datasets import points_utils

RNG = 4.8
DELTAS = [0.0, 0.5, 1.0]


def load(cfg_path, ckpt, dev):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model).to(dev).eval()
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)
    return cfg, model


def crop_inputs(tr, fid, ref, pcr, dev):
    prev = points_utils.crop_pc_in_range(tr[fid - 1]['pc'], ref, pcr).points.T
    this = points_utils.crop_pc_in_range(tr[fid]['pc'], ref, pcr).points.T
    if prev.shape[0] < 1 or this.shape[0] < 1:
        return None
    return {'prev_points': [torch.as_tensor(prev, dtype=torch.float32, device=dev)],
            'this_points': [torch.as_tensor(this, dtype=torch.float32, device=dev)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gated-config', required=True); ap.add_argument('--gated-ckpt', required=True)
    ap.add_argument('--base-config', required=True); ap.add_argument('--base-ckpt', required=True)
    ap.add_argument('--split', default='Valid'); ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='mech/')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'
    rng = np.random.RandomState(0)

    gcfg, gmodel = load(args.gated_config, args.gated_ckpt, dev)
    bcfg, bmodel = load(args.base_config, args.base_ckpt, dev)
    pcr = gcfg.point_cloud_range
    head = gmodel.head
    assert hasattr(head, 'gate'), 'mech_pathway expects a gated head'

    dcfg = dict(gcfg.test_dataloader['dataset'])
    dcfg['dataset'] = dict(dcfg['dataset']); dcfg['dataset']['split'] = args.split
    ds = DATASETS.build(dcfg)
    samples = getattr(ds, 'training_samples', None) or getattr(ds, 'tracklets', None) \
        or [ds[i] for i in range(len(ds))]

    dirs = {}
    for ti, tr in enumerate(samples):
        for fid in range(1, len(tr)):
            a = rng.uniform(0, 2 * np.pi)
            dirs[(ti, fid)] = np.array([np.cos(a), np.sin(a)])

    rows = []  # per (frame, delta)
    with torch.no_grad():
        for ti, tr in enumerate(samples):
            for fid in range(1, len(tr)):
                ref0 = tr[fid - 1]['3d_bbox']; cur = tr[fid]['3d_bbox']
                d0 = ref0.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref0.center))
                mag0 = float(np.linalg.norm(d0[:2]))
                if mag0 > RNG:
                    continue
                u = dirs[(ti, fid)]
                for delta in DELTAS:
                    ref = points_utils.getOffsetBB(
                        ref0, [delta * u[0], delta * u[1], 0.0, 0.0],
                        degrees=True, use_z=True, limit_box=False) if delta > 0 else ref0
                    d = ref.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref.center))
                    gt = np.array([float(d[0]), float(d[1])])
                    inp = crop_inputs(tr, fid, ref, pcr, dev)
                    if inp is None:
                        continue
                    feats = gmodel.backbone(inp['prev_points'] + inp['this_points'])
                    fused = gmodel.fuse(feats)
                    disp = fused['disp']; feat = fused['feat']; conf = fused['conf']
                    o = head.reg(torch.cat([feat, disp, conf], 1))
                    r = o[:, :2]
                    g = torch.sigmoid(head.gate(torch.cat([feat, conf], 1)))
                    dcorr = disp[0].cpu().numpy(); rr = r[0].cpu().numpy(); gg = float(g[0, 0])
                    # baseline model error at same ref
                    bcoors = bmodel.inference(inp)
                    rows.append(dict(
                        ti=ti, fid=fid, delta=delta, mag0=mag0,
                        gt=gt.tolist(), dcorr=dcorr.tolist(), r=rr.tolist(),
                        g=gg, conf=float(conf[0, 0]),
                        err_base=float(np.linalg.norm(bcoors[:2] - gt))))

    # apply injection modes offline (shuffle within delta x stratum)
    rows_np = rows
    disp_pool = {}
    for row in rows_np:
        key = (row['delta'], row['mag0'] < 0.5)
        disp_pool.setdefault(key, []).append(np.array(row['dcorr']))
    shuf = {k: np.array(v)[rng.permutation(len(v))] for k, v in disp_pool.items()}
    ptr = {k: 0 for k in shuf}
    for row in rows_np:
        gt = np.array(row['gt']); r = np.array(row['r']); g = row['g']; dcorr = np.array(row['dcorr'])
        key = (row['delta'], row['mag0'] < 0.5)
        d_shuf = shuf[key][ptr[key]]; ptr[key] += 1
        row['err_a1'] = float(np.linalg.norm(r + g * dcorr - gt))
        row['err_a0'] = float(np.linalg.norm(r - gt))
        row['err_shuf'] = float(np.linalg.norm(r + g * d_shuf - gt))
        row['err_oracle'] = float(np.linalg.norm(r + g * gt - gt))
        row['corr_err'] = float(np.linalg.norm(dcorr - gt))
        row['inj_norm'] = float(g * np.linalg.norm(dcorr))

    # summarise per delta x stratum
    summ = {}
    for delta in DELTAS:
        for st, lo, hi in [('low', -1, 0.5), ('high', 0.5, RNG + 1)]:
            sel = [x for x in rows_np if x['delta'] == delta and lo <= x['mag0'] < hi]
            if not sel:
                continue
            m = lambda k: float(np.mean([x[k] for x in sel]))
            summ[f'd{delta}_{st}'] = dict(
                n=len(sel), **{k: m('err_' + k) for k in ['base', 'a1', 'a0', 'shuf', 'oracle']},
                corr_err=m('corr_err'), gate=m('g'), inj_norm=m('inj_norm'),
                deficit_closure=((m('err_a1') - m('err_a0')) / (m('err_a1') - m('err_base'))
                                 if abs(m('err_a1') - m('err_base')) > 1e-6 else None))
    json.dump({'tag': args.tag, 'split': args.split, 'summary': summ}, open(f'{args.out}/{args.tag}.json', 'w'), indent=2)
    np.savez(f'{args.out}/{args.tag}_raw.npz',
             **{k: np.array([x[k] for x in rows_np]) for k in
                ['ti', 'fid', 'delta', 'mag0', 'g', 'conf', 'corr_err', 'inj_norm',
                 'err_base', 'err_a1', 'err_a0', 'err_shuf', 'err_oracle']})
    print(f'{args.tag} [{args.split}]')
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
