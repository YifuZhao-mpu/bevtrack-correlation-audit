"""Extract framework-figure assets from a real frame: the point-cloud crops the
model sees, the BEV feature maps it builds, and the correlation response surface
it correlates -- the object the whole paper is about and which no figure shows.

Each asset is saved on its own, transparent background, no axes, so they can be
composed into a diagram. A contact sheet is written alongside for picking.

  python tools/make_assets.py --config configs/kitti/m_gated_s0.py \
      --ckpt work_dir/m_gated_s0/epoch_500.pth --tracklet 3 --frame 12 --tag gated
"""
import argparse, copy, json, os, sys
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf, np.NINF = np.inf, -np.inf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets, models  # noqa: F401  (registers)
from datasets import points_utils
from models.corr_head import FEATURE_STRIDE_M

OUT = 'paper/pr/figures/assets'


def save(fig, name, tag):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{tag}_{name}.{ext}', dpi=220, transparent=True,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('  wrote', f'{tag}_{name}')


def bare(figsize=(2.6, 2.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tracklet', type=int, default=3); ap.add_argument('--frame', type=int, default=12)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--scan', type=int, default=0,
                    help='instead of writing assets, report matcher error for the '
                         'first N tracklets so a representative frame can be chosen')
    a = ap.parse_args()

    cfg = Config.fromfile(a.config)
    dcfg = copy.deepcopy(cfg.test_dataloader['dataset'])
    dcfg['dataset']['preloading'] = False
    ds = DATASETS.build(dcfg)
    model = MODELS.build(cfg.model).cuda().eval()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck.get('state_dict', ck), strict=False)

    if a.scan:
        rows = []
        with torch.no_grad():
            for ti in range(min(a.scan, len(ds))):
                tr = ds[ti]
                for fid in range(1, len(tr)):
                    ref, cur = tr[fid - 1]['3d_bbox'], tr[fid]['3d_bbox']
                    dd = ref.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref.center))
                    g = np.array([float(dd[0]), float(dd[1])])
                    mag = float(np.linalg.norm(g))
                    if not (0.5 <= mag <= 2.0):
                        continue
                    pv = points_utils.crop_pc_in_range(tr[fid - 1]['pc'], ref, cfg.point_cloud_range).points.T
                    th = points_utils.crop_pc_in_range(tr[fid]['pc'], ref, cfg.point_cloud_range).points.T
                    if pv.shape[0] < 60 or th.shape[0] < 60:
                        continue
                    pl = [torch.as_tensor(pv, dtype=torch.float32, device='cuda'),
                          torch.as_tensor(th, dtype=torch.float32, device='cuda')]
                    fe = model.backbone(pl)
                    fu = model.fuse
                    dsp, cf, _ = fu.correlate(fu.embed(fe[0:1]), fu.embed(fe[1:2]))
                    e = float(np.linalg.norm([float(dsp[0, 0]) - g[0], float(dsp[0, 1]) - g[1]]))
                    rows.append((e, ti, fid, mag, int(pv.shape[0]), float(cf[0, 0])))
        rows.sort()
        print(f'{"err":>7s} {"trk":>4s} {"frm":>4s} {"|d|":>6s} {"npts":>5s} {"conf":>6s}')
        for r in rows[:8]:
            print(f'{r[0]:7.3f} {r[1]:4d} {r[2]:4d} {r[3]:6.3f} {r[4]:5d} {r[5]:6.3f}')
        return

    tr = ds[a.tracklet]
    fid = a.frame
    ref, cur = tr[fid - 1]['3d_bbox'], tr[fid]['3d_bbox']
    d = ref.rotation_matrix.T @ (np.asarray(cur.center) - np.asarray(ref.center))
    gt = np.array([float(d[0]), float(d[1])])
    rng = cfg.point_cloud_range
    prev = points_utils.crop_pc_in_range(tr[fid - 1]['pc'], ref, rng).points.T
    this = points_utils.crop_pc_in_range(tr[fid]['pc'], ref, rng).points.T
    print(f'tracklet {a.tracklet} frame {fid}: |d| = {np.linalg.norm(gt):.3f} m, '
          f'{prev.shape[0]}/{this.shape[0]} points')

    # (1,2) the two point-cloud crops, top-down
    for pts, nm in ((prev, 'pc_prev'), (this, 'pc_curr')):
        fig, ax = bare()
        ax.scatter(pts[:, 0], pts[:, 1], s=1.6, c='#2a78d6', alpha=.55, linewidths=0)
        ax.set_xlim(rng[0], rng[3]); ax.set_ylim(rng[1], rng[4]); ax.set_aspect('equal')
        save(fig, nm, a.tag)

    with torch.no_grad():
        pl = [torch.as_tensor(prev, dtype=torch.float32, device='cuda'),
              torch.as_tensor(this, dtype=torch.float32, device='cuda')]
        feats = model.backbone(pl)
        fuser = model.fuse
        pe, te = fuser.embed(feats[0:1]), fuser.embed(feats[1:2])
        disp, conf, heat = fuser.correlate(pe, te)

    # (3,4) the BEV feature maps the encoder produces
    for f, nm in ((feats[0], 'bev_prev'), (feats[1], 'bev_curr')):
        m = f.mean(0).float().cpu().numpy()
        fig, ax = bare()
        ax.imshow(m, origin='lower', cmap='viridis')
        save(fig, nm, a.tag)

    # (5) the correlation response surface, with the decode and the truth on it
    h = heat[0].float().cpu().numpy()
    H, W = h.shape
    q = H // 2
    ext = [(1 - q - .5) * FEATURE_STRIDE_M, (W - 1 - q + .5) * FEATURE_STRIDE_M,
           (1 - q - .5) * FEATURE_STRIDE_M, (H - 1 - q + .5) * FEATURE_STRIDE_M]
    fig, ax = bare((3.0, 2.8))
    im = ax.imshow(h[1:, 1:], origin='lower', extent=ext, cmap='magma', aspect='equal')
    dx, dy = float(disp[0, 0]), float(disp[0, 1])
    ax.plot([dx], [dy], 'o', ms=9, mfc='none', mec='#1baf7a', mew=2.0, label='decoded')
    ax.plot([gt[0]], [gt[1]], '+', ms=12, mec='#eb6834', mew=2.2, label='ground truth')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='none', mfc='none', mec='#1baf7a',
                              mew=2.0, ms=9, label='decoded'),
                       Line2D([], [], marker='+', ls='none', mec='#eb6834', mew=2.2,
                              ms=12, label='ground truth')],
              fontsize=7, frameon=False, labelcolor='white', loc='upper left',
              handletextpad=0.6, labelspacing=0.5)
    save(fig, 'corr_heat', a.tag)
    print(f'  decoded ({dx:+.3f},{dy:+.3f})  gt ({gt[0]:+.3f},{gt[1]:+.3f})  '
          f'err {np.linalg.norm([dx-gt[0], dy-gt[1]]):.3f} m  conf {float(conf[0,0]):.3f}')

    json.dump(dict(config=a.config, ckpt=a.ckpt, tracklet=a.tracklet, frame=fid,
                   gt=gt.tolist(), decoded=[dx, dy], conf=float(conf[0, 0]),
                   err=float(np.linalg.norm([dx - gt[0], dy - gt[1]])),
                   n_prev=int(prev.shape[0]), n_curr=int(this.shape[0])),
              open(f'{OUT}/{a.tag}_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
