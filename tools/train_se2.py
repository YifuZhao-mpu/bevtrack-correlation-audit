"""M1 harness (epoch-based, Codex protocol): train one head (implicit|se2trans|se2full)
on a category with the pillar/GroupNorm encoder; eval by displacement strata every N
epochs; select best checkpoint by in-band Success from epoch>=select_after. Same
encoder/data/budget for both heads -> head is the controlled variable.
Strata: static<0.5m, in-band 0.5-4.8m, out-of-crop>4.8m.
"""
import argparse, sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
from mmengine.config import ConfigDict
from datasets.nuscenes import NuScenesDataset
from datasets.sampler import TrainSampler
from datasets import points_utils
from datasets.metrics import estimateOverlap, estimateAccuracy
from models.voxelnet import VoxelNet
from models.bevfuser import BEVFuser
from models.simple_head import SimpleHead
from models.se2_head import SE2Head, IdentityFuser

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
PCR = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
DATA = '/home/zyf/data/nuscenes'
dev = 'cuda'


def build_model(head, w_static=0.2, impl_disp_weighted=False, weight_scheme='none', focal_gamma=2.0):
    backbone = VoxelNet(PCR, [0.075, 0.075, 0.15], [21, 128, 128])
    if head == 'implicit':
        fuse, h = BEVFuser(), SimpleHead(disp_weighted=impl_disp_weighted, w_static=w_static,
                                         weight_scheme=weight_scheme, focal_gamma=focal_gamma)
    elif head == 'se2trans':
        fuse, h = IdentityFuser(), SE2Head(translation_only=True, up=4, kt=21, pcr=(-4.8, 4.8))
    elif head == 'heatmap':
        from models.heatmap_head import HeatmapHead
        fuse, h = IdentityFuser(), HeatmapHead(up=4, kt=21, pcr=(-4.8, 4.8), w_static=w_static)
    elif head == 'se2full':
        fuse, h = IdentityFuser(), SE2Head(translation_only=False, n_theta=15, theta_max=28.0, up=4, kt=21, pcr=(-4.8, 4.8))
    else:
        raise ValueError(head)
    return torch.nn.ModuleDict({'b': backbone, 'f': fuse, 'h': h})


def collate(batch):
    return {'prev': [b['inputs']['prev_points'] for b in batch],
            'this': [b['inputs']['this_points'] for b in batch],
            'label': torch.stack([b['data_samples']['box_label'] for b in batch])}


def fwd(model, prev, this):
    feats = model['b']([p.to(dev) for p in prev] + [t.to(dev) for t in this])
    return model['h'](model['f'](feats))


def succ_prec(ovs, dis):
    if len(ovs) == 0:
        return {'success': 0.0, 'precision': 0.0, 'n': 0}
    ov = torch.tensor(ovs); di = torch.tensor(dis)
    sx = torch.linspace(0, 1, 21); px = torch.linspace(0, 2, 21)
    succ = torch.tensor([(ov >= t).float().mean() for t in sx])
    prec = torch.tensor([(di <= t).float().mean() for t in px])
    return {'success': float(torch.trapz(succ, sx) * 100),
            'precision': float(torch.trapz(prec, px) * 100 / 2), 'n': len(ovs)}


@torch.no_grad()
def evaluate(model, val_ds, max_tracklets=None, lo=0.5, hi=4.8):
    model.eval()
    bk = {k: ([], []) for k in ('static', 'inband', 'outcrop', 'all')}
    n = val_ds.get_num_tracklets()
    idxs = list(range(n if not max_tracklets else min(n, max_tracklets)))
    for ti in idxs:
        nf = val_ds.get_num_frames_tracklet(ti)
        frames = val_ds.get_frames(ti, list(range(nf)))
        results = [frames[0]['3d_bbox']]
        for fi in range(1, nf):
            ref = results[-1]
            pp = points_utils.crop_pc_in_range(frames[fi - 1]['pc'], ref, PCR).points.T
            tp = points_utils.crop_pc_in_range(frames[fi]['pc'], ref, PCR).points.T
            if pp.shape[0] < 1: pp = np.zeros((1, 3), 'float32')
            if tp.shape[0] < 1: tp = np.zeros((1, 3), 'float32')
            with torch.autocast('cuda', dtype=torch.float16):
                res = fwd(model, [torch.as_tensor(pp, dtype=torch.float32)],
                          [torch.as_tensor(tp, dtype=torch.float32)])
            coors = res['coors'][0].float().cpu().numpy()
            pred = points_utils.getOffsetBB(ref, coors, degrees=True, use_z=True, limit_box=False)
            results.append(pred)
            gt, gtp = frames[fi]['3d_bbox'], frames[fi - 1]['3d_bbox']
            ov = estimateOverlap(gt, pred, dim=3, up_axis=[0, 0, 1])
            di = estimateAccuracy(gt, pred, dim=3, up_axis=[0, 0, 1])
            disp = float(np.linalg.norm(gt.center[:2] - gtp.center[:2]))
            b = 'static' if disp < lo else ('inband' if disp <= hi else 'outcrop')
            for k in (b, 'all'):
                bk[k][0].append(ov); bk[k][1].append(di)
    model.train()
    return {k: succ_prec(*v) for k, v in bk.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True, choices=['implicit', 'se2trans', 'se2full', 'heatmap'])
    ap.add_argument('--category', default='Car')
    ap.add_argument('--max_epochs', type=int, default=30)
    ap.add_argument('--eval_every_epochs', type=int, default=2)
    ap.add_argument('--select_after_epoch', type=int, default=10)
    ap.add_argument('--micro_batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-4)
    ap.add_argument('--clip', type=float, default=5.0)
    ap.add_argument('--warmup_steps', type=int, default=500)
    ap.add_argument('--eval_tracklets', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--w_static', type=float, default=0.2,
                    help='disp-weight for static (<0.5m) frames in heatmap loss (and implicit if --impl_disp_weighted)')
    ap.add_argument('--impl_disp_weighted', action='store_true',
                    help='apply the same 0.2/1.0/0.0 disp weighting to the implicit RLE loss (symmetric-objective baseline)')
    ap.add_argument('--weight_scheme', default='none', choices=['none','disp','invfreq','harderr','focal'],
                    help='implicit-head loss weighting scheme (E2 competing baselines; overrides --impl_disp_weighted)')
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or f'/home/zyf/paper/refine-logs/m1_{args.head}_{args.category}'
    os.makedirs(out, exist_ok=True)
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    def _wif(wid):
        s = args.seed * 1000 + wid
        np.random.seed(s % 2**31); random.seed(s)
    gen = torch.Generator(); gen.manual_seed(args.seed)

    print(f"[{args.head}/{args.category}] loading data (preload)...", flush=True)
    tcfg = ConfigDict(dict(num_candidates=2, target_thr=None, search_thr=5,
                           point_cloud_range=PCR, time_flip=True, flip=True))
    train = TrainSampler(dataset=dict(type='NuScenesDataset', path=DATA, split='train_track',
                                      category_name=args.category, preloading=True, preload_offset=10), cfg=tcfg)
    val = NuScenesDataset(path=DATA, split='val', category_name=args.category, preloading=True)
    dl = DataLoader(train, batch_size=args.micro_batch, shuffle=True, num_workers=6,
                    collate_fn=collate, drop_last=True, persistent_workers=True,
                    generator=gen, worker_init_fn=_wif)
    spe = len(dl)
    print(f"  train samples={len(train)} steps/epoch={spe} | val tracklets={val.get_num_tracklets()}", flush=True)

    model = build_model(args.head, w_static=args.w_static,
                        impl_disp_weighted=args.impl_disp_weighted,
                        weight_scheme=args.weight_scheme, focal_gamma=args.focal_gamma).to(dev).train()
    print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    step, t0, best_in, best_epoch = 0, time.time(), -1, -1
    horizon = args.max_epochs * spe  # cosine horizon
    log = []
    for epoch in range(1, args.max_epochs + 1):
        for batch in dl:
            # matched schedule for BOTH heads: linear warmup -> cosine decay to min_lr
            if step < args.warmup_steps:
                cur_lr = args.lr * (step + 1) / max(1, args.warmup_steps)
            else:
                tt = min(1.0, (step - args.warmup_steps) / max(1, horizon - args.warmup_steps))
                cur_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * tt))
            for g in opt.param_groups:
                g['lr'] = cur_lr
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                res = fwd(model, batch['prev'], batch['this'])
                gt = batch['label'].to(dev)
                ld = model['h'].loss(res, {'box_label': list(gt)})
                loss = sum(ld.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if torch.isfinite(gnorm):
                scaler.step(opt)
            scaler.update()
            step += 1
            if step % 200 == 0:
                print(f"  ep{epoch} step {step:6d} loss {loss.item():7.3f} {step/(time.time()-t0):.1f}it/s", flush=True)
        if epoch % args.eval_every_epochs == 0:
            ev = evaluate(model, val, max_tracklets=args.eval_tracklets)
            print(f"[EVAL ep{epoch}] all {ev['all']['success']:.2f}/{ev['all']['precision']:.2f} "
                  f"static {ev['static']['success']:.2f}(n{ev['static']['n']}) "
                  f"inband {ev['inband']['success']:.2f}/{ev['inband']['precision']:.2f}(n{ev['inband']['n']}) "
                  f"out {ev['outcrop']['success']:.2f}(n{ev['outcrop']['n']})", flush=True)
            log.append({'epoch': epoch, 'step': step, **ev})
            json.dump(log, open(f'{out}/eval_log.json', 'w'), indent=2)
            if epoch >= args.select_after_epoch and ev['inband']['success'] > best_in:
                best_in, best_epoch = ev['inband']['success'], epoch
                torch.save(model.state_dict(), f'{out}/best.pt')
    print(f"[best epoch {best_epoch} inband {best_in:.2f}] running full-val eval on best ckpt", flush=True)
    if os.path.exists(f'{out}/best.pt'):
        model.load_state_dict(torch.load(f'{out}/best.pt'))
    ev = evaluate(model, val, max_tracklets=None)
    print("FINAL:", json.dumps(ev), flush=True)
    json.dump({'final': ev, 'best_epoch': best_epoch, 'log': log, 'args': vars(args)},
              open(f'{out}/final.json', 'w'), indent=2)
    print(f"DONE {args.head} {args.category}", flush=True)


if __name__ == '__main__':
    main()
