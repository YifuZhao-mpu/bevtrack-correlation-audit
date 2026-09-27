"""M2 diagnostic: WHY does the learned BEV correlation lose to implicit regression?

On KITTI Car test moving/static frames with ORACLE ref (prev GT box), measure:
  (a) learned corr-heat displacement recall (argmax / top-5) at the 16x16 feature
      resolution (0.60 m cells) -- vs the FFT occupancy oracle (premise: 75.2% top-5
      at 128x128 / +-0.15 m);
  (b) soft-argmax error vs argmax error (multimodal-mean pollution);
  (c) both vs the zero-motion baseline |gt| (is the corr branch anti-informative?);
  (d) conf (peak prob) vs error correlation (could gating work in principle?).
Writes diag_corr.json. Run: CUDA_VISIBLE_DEVICES=0 python tools/diag_corr_learned.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
import datasets  # registers
import models    # registers
from datasets import points_utils
from models.corr_head import decode_corr_heat, FEATURE_STRIDE_M

# Config and checkpoint are CLI arguments. The superseded version hardcoded
# work_dir/car_corr2/best_precision_epoch_450.pth -- a pre-decode-fix v2 model,
# read at a checkpoint selected by Precision, i.e. exactly the practice this
# project criticises. Point it at a terminal checkpoint of a matched arm.
R128, RNG = 128, 4.8
VOX128 = 2 * RNG / R128


def bev_occ(points):
    g = np.zeros((R128, R128), np.float32)
    if points.shape[0] < 2:
        return g
    ix = ((points[:, 0] + RNG) / VOX128).astype(int)
    iy = ((points[:, 1] + RNG) / VOX128).astype(int)
    m = (ix >= 0) & (ix < R128) & (iy >= 0) & (iy < R128)
    g[iy[m], ix[m]] = 1.0
    return g


def fft_peaks(prev_pts, this_pts, k=5):
    tmpl = prev_pts[(np.abs(prev_pts[:, 0]) < 2.0) & (np.abs(prev_pts[:, 1]) < 2.0)]
    pb, tb = bev_occ(tmpl), bev_occ(this_pts)
    if pb.sum() < 3 or tb.sum() < 3:
        return None
    corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(tb) * np.conj(np.fft.fft2(pb)))))
    flat = np.argsort(-corr.ravel())[:k]
    out = []
    for idx in flat:
        py, px = np.unravel_index(idx, corr.shape)
        out.append(((px - R128 // 2) * VOX128, (py - R128 // 2) * VOX128))  # (dx,dy) meters
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-preload', action='store_true')
    args = ap.parse_args()
    cfg = Config.fromfile(args.config)
    if args.no_preload:
        cfg.test_dataloader['dataset']['dataset']['preloading'] = False
    dev = 'cuda'
    model = MODELS.build(cfg.model).to(dev).eval()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    print('load_state_dict missing:', missing, ' unexpected:', unexpected)

    ds = DATASETS.build(cfg.test_dataloader['dataset'])
    samples = getattr(ds, 'training_samples', None) or getattr(ds, 'tracklets', None)
    if samples is None:
        samples = [ds[i] for i in range(len(ds))]

    fuser = model.fuse
    Hf = None  # feature grid size, discovered on first forward
    grid = None

    rec = {'static': [], 'inband': []}   # per-frame dicts
    n_skip = 0
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
                if prev_pts.shape[0] < 3 or this_pts.shape[0] < 3:
                    n_skip += 1
                    continue
                pl = [torch.as_tensor(prev_pts, dtype=torch.float32, device=dev),
                      torch.as_tensor(this_pts, dtype=torch.float32, device=dev)]
                feats = model.backbone(pl)                       # (2,128,Hf,Hf)
                pe = fuser.embed(feats[0:1]); te = fuser.embed(feats[1:2])
                B, C, H, W = pe.shape
                if Hf is None:
                    Hf = H
                tgt = pe[:, :, H // 2, W // 2]
                heat = torch.einsum('bc,bchw->bhw', tgt, te) / (C ** 0.5)
                # Decode exactly as the tracker does: query-relative soft-argmax on the
                # symmetric lag support, at the true feature stride. The superseded
                # version of this script built a linspace(-RNG, RNG, H) grid, whose
                # zero lag sits at (+0.32, +0.32) m and whose spacing is 9.6/(H-1) =
                # 0.64 m rather than the 0.60 m stride; that bias is what produced the
                # ~0.45 m "floor" reported in the pre-2026-07-18 diagnostics.
                disp_t, conf_t = decode_corr_heat(heat, FEATURE_STRIDE_M)   # heat is (1,H,W)
                sa_dx, sa_dy = float(disp_t[0, 0]), float(disp_t[0, 1])
                conf = float(conf_t[0, 0])
                # argmax + top-5, on the same symmetric support and the same lag axes
                hs = heat[0, 1:, 1:]
                qy, qx = H // 2, W // 2
                flat = torch.topk(hs.reshape(-1), 5).indices.cpu().numpy()
                Ws = W - 1
                peaks = [((i % Ws) + 1, (i // Ws) + 1) for i in flat]   # (ix,iy) into heat
                peaks_m = [((ix - qx) * FEATURE_STRIDE_M, (iy - qy) * FEATURE_STRIDE_M)
                           for ix, iy in peaks]
                cell = FEATURE_STRIDE_M
                err_am = float(np.linalg.norm(np.array(peaks_m[0]) - gt))
                err_sa = float(np.linalg.norm(np.array([sa_dx, sa_dy]) - gt))
                hit1 = err_am <= cell
                hit5 = any(np.linalg.norm(np.array(p) - gt) <= cell for p in peaks_m)
                # FFT oracle on raw occupancy (moving frames only, to mirror premise)
                fp = fft_peaks(prev_pts, this_pts) if stratum == 'inband' else None
                if fp is None:
                    f1_15 = f5_15 = f1_64 = f5_64 = None
                else:
                    e = [np.linalg.norm(np.array(p) - gt) for p in fp]
                    f1_15, f5_15 = e[0] <= 0.15 + 1e-9, min(e) <= 0.15 + 1e-9
                    f1_64, f5_64 = e[0] <= cell, min(e) <= cell
                rec[stratum].append(dict(mag=mag, err_am=err_am, err_sa=err_sa, conf=conf,
                                         hit1=hit1, hit5=hit5,
                                         f1_15=f1_15, f5_15=f5_15, f1_64=f1_64, f5_64=f5_64))

    out = {'feature_grid': Hf, 'cell_m': FEATURE_STRIDE_M, 'skipped': n_skip,
           'decode': 'query-relative, corrected 2026-09-23'}
    for s in ('static', 'inband'):
        rows = rec[s]
        if not rows:
            continue
        A = {k: np.array([r[k] for r in rows]) for k in rows[0] if rows[0][k] is not None or k.startswith('f')}
        mag = np.array([r['mag'] for r in rows])
        conf = np.array([r['conf'] for r in rows])
        err_sa = np.array([r['err_sa'] for r in rows])
        res = dict(
            n=len(rows), mean_gt_disp=float(mag.mean()),
            zero_motion_err=float(mag.mean()),
            argmax_err=float(np.mean([r['err_am'] for r in rows])),
            softargmax_err=float(err_sa.mean()),
            softargmax_worse_than_zero_pct=float(100 * np.mean(err_sa > mag)),
            recall_top1_1cell=float(100 * np.mean([r['hit1'] for r in rows])),
            recall_top5_1cell=float(100 * np.mean([r['hit5'] for r in rows])),
        )
        # conf-error relation: mean soft-argmax error in bottom vs top conf quartile
        q1, q3 = np.quantile(conf, 0.25), np.quantile(conf, 0.75)
        res['err_lowconf_q'] = float(err_sa[conf <= q1].mean())
        res['err_highconf_q'] = float(err_sa[conf >= q3].mean())
        fmask = [r['f1_15'] is not None for r in rows]
        if any(fmask):
            for k in ('f1_15', 'f5_15', 'f1_64', 'f5_64'):
                res['fft_' + k] = float(100 * np.mean([r[k] for r, m in zip(rows, fmask) if m]))
        out[s] = res

    print(json.dumps(out, indent=2))
    json.dump(out, open(args.out, 'w'), indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
