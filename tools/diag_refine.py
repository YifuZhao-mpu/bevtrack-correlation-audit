"""Pre-launch validation for v4b's local-refinement decode.

On the TRAINED corr-v3 checkpoint (oracle ref, KITTI Car test), decode the same
correlation heat four ways and compare per-frame displacement error:
  softargmax : global soft-argmax (v3 shipping decode; ~0.45m static floor)
  argmax     : peak cell center (pure quantization reference)
  refine     : 3x3 local softmax centroid around the peak (v4b decode)
  refine_tX  : same with temperature X on the local logits (in case peaks are
               too sharp and the local softmax collapses to one-hot)
GO if refine static err clearly < softargmax static err (~breaks the floor).
Run: CUDA_VISIBLE_DEVICES=1 python tools/diag_refine.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
import datasets  # noqa: registers
import models    # noqa: registers
from datasets import points_utils

CKPT = 'work_dir/car_corr2/best_precision_epoch_450.pth'
RNG = 4.8


def refine(heat, temp=1.0):
    """3x3 local softmax centroid around argmax -> (dx, dy) meters."""
    H, W = heat.shape
    idx = int(heat.reshape(-1).argmax())
    y0 = min(max(idx // W, 1), H - 2)
    x0 = min(max(idx % W, 1), W - 2)
    patch = heat[y0 - 1:y0 + 2, x0 - 1:x0 + 2] / temp
    pw = F.softmax(patch.reshape(-1), 0).reshape(3, 3)
    off = torch.tensor([-1., 0., 1.], device=heat.device)
    cy = y0 + float((pw.sum(1) * off).sum())
    cx = x0 + float((pw.sum(0) * off).sum())
    step = 2 * RNG / (H - 1)
    return np.array([-RNG + cx * step, -RNG + cy * step])


def main():
    cfg = Config.fromfile('configs/kitti/car_corr2.py')
    dev = 'cuda'
    model = MODELS.build(cfg.model).to(dev).eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)

    ds = DATASETS.build(cfg.test_dataloader['dataset'])
    samples = getattr(ds, 'training_samples', None) or getattr(ds, 'tracklets', None)
    if samples is None:
        samples = [ds[i] for i in range(len(ds))]

    fuser = model.fuse
    grid = None
    errs = {'static': {}, 'inband': {}}
    decoders = ['softargmax', 'argmax', 'refine', 'refine_t2', 'refine_t4']
    for s in errs:
        errs[s] = {d: [] for d in decoders}

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
                    continue
                pl = [torch.as_tensor(prev_pts, dtype=torch.float32, device=dev),
                      torch.as_tensor(this_pts, dtype=torch.float32, device=dev)]
                feats = model.backbone(pl)
                pe = fuser.embed(feats[0:1]); te = fuser.embed(feats[1:2])
                B, C, H, W = pe.shape
                if grid is None:
                    grid = torch.linspace(-RNG, RNG, H, device=dev)
                tgt = pe[:, :, H // 2, W // 2]
                heat = (torch.einsum('bc,bchw->bhw', tgt, te) / (C ** 0.5))[0]
                prob = F.softmax(heat.reshape(-1), 0).reshape(H, W)
                sa = np.array([float((prob.sum(0) * grid).sum()),
                               float((prob.sum(1) * grid).sum())])
                idx = int(heat.reshape(-1).argmax())
                am = np.array([float(grid[idx % W]), float(grid[idx // W])])
                dec = {'softargmax': sa, 'argmax': am,
                       'refine': refine(heat), 'refine_t2': refine(heat, 2.0),
                       'refine_t4': refine(heat, 4.0)}
                for k, v in dec.items():
                    errs[stratum][k].append(float(np.linalg.norm(v - gt)))

    out = {}
    for s in errs:
        out[s] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v))
                  for k, v in errs[s].items() if v}
    print(json.dumps(out, indent=2))
    json.dump(out, open('diag_refine.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
