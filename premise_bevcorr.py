"""BEVTrack-Motion premise (NO training): can explicit BEV cross-correlation between the
prev-target crop and current-search crop RECOVER the GT displacement on moving frames?
Oracle ref = prev GT box. If even with oracle ref top-5 recall <55%, explicit displacement
search is dead. GO if top-5 GT-displacement recall >=70% (gpt-5.5's gate).
"""
import os, sys, numpy as np
from mmengine.config import Config
from mmengine.registry import DATASETS
import datasets  # registers KittiDataset
from datasets import points_utils

R, RANGE = 128, 4.8           # BEV grid 128x128 over +/-4.8m (BEVTrack config)
VOX = 2 * RANGE / R


def bev(points):              # points (N,3) in ref-box coords -> 128x128 occupancy
    g = np.zeros((R, R), np.float32)
    if points.shape[0] < 2:
        return g
    ix = ((points[:, 0] + RANGE) / VOX).astype(int)
    iy = ((points[:, 1] + RANGE) / VOX).astype(int)
    m = (ix >= 0) & (ix < R) & (iy >= 0) & (iy < R)
    g[iy[m], ix[m]] = 1.0
    return g


def topk_shifts(corr, k=5):   # top-k peaks of fftshifted corr -> list of (dy,dx) cell shifts
    flat = np.argsort(-corr.ravel())[:k]
    out = []
    for idx in flat:
        py, px = np.unravel_index(idx, corr.shape)
        out.append((py - R // 2, px - R // 2))
    return out


def main():
    cfg = Config.fromfile('configs/kitti/car.py')
    ds = DATASETS.build(cfg.test_dataloader['dataset'])
    tracklets = ds.dataset if hasattr(ds, 'dataset') else ds
    # KittiDataset preloads tracklets as list-of-frames; find the accessor
    samples = getattr(ds, 'training_samples', None) or getattr(ds, 'tracklets', None)
    if samples is None:
        samples = [ds[i] for i in range(min(len(ds), 200))]
    n_move = 0; top1 = 0; top5 = 0; disps = []
    for tr in samples:
        for fid in range(1, len(tr)):
            ref = tr[fid - 1]['3d_bbox']; cur = tr[fid]['3d_bbox']
            d = ref.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref.center))
            dx, dy = float(d[0]), float(d[1])
            mag = (dx**2 + dy**2) ** 0.5
            if mag < 0.5 or mag > RANGE:       # moving, in-crop only
                continue
            prev_pts = points_utils.crop_pc_in_range(tr[fid-1]['pc'], ref, cfg.point_cloud_range).points.T
            this_pts = points_utils.crop_pc_in_range(tr[fid]['pc'], ref, cfg.point_cloud_range).points.T
            # TEMPLATE = target only (points within the prev box footprint, ~+/-2m), so the
            # static background doesn't dominate the correlation. SEARCH = full crop.
            tmpl = prev_pts[(np.abs(prev_pts[:, 0]) < 2.0) & (np.abs(prev_pts[:, 1]) < 2.0)]
            prev_bev = bev(tmpl)
            this_bev = bev(this_pts)
            if prev_bev.sum() < 3 or this_bev.sum() < 3:
                continue
            F1 = np.fft.fft2(prev_bev); F2 = np.fft.fft2(this_bev)
            corr = np.fft.fftshift(np.real(np.fft.ifft2(F2 * np.conj(F1))))
            gt_cell = (int(round(dy / VOX)), int(round(dx / VOX)))   # (dy,dx) in cells
            peaks = topk_shifts(corr, 5)
            hit1 = (abs(peaks[0][0]-gt_cell[0]) <= 2 and abs(peaks[0][1]-gt_cell[1]) <= 2)
            hit5 = any(abs(p[0]-gt_cell[0]) <= 2 and abs(p[1]-gt_cell[1]) <= 2 for p in peaks)
            n_move += 1; top1 += int(hit1); top5 += int(hit5); disps.append(mag)
    print(f"moving frames evaluated: {n_move}")
    print(f"mean GT displacement: {np.mean(disps):.2f} m")
    print(f"top-1 GT-displacement recall: {100*top1/max(n_move,1):.1f}%")
    print(f"top-5 GT-displacement recall: {100*top5/max(n_move,1):.1f}%")
    print("\nGATE: GO if top-5 >=70%  |  KILL if top-5 <55%")


if __name__ == '__main__':
    main()
