"""Codex Attempt-B: same as Attempt-A but BALANCED batch sampling (50% static / 50% in-band,
out-of-crop excluded) via WeightedRandomSampler -> ~3x more displaced-target gradient.
Last cosine attempt; if in-band still doesn't clear, pivot to learned-heatmap head."""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from mmengine.config import ConfigDict
from datasets.nuscenes import NuScenesDataset
from datasets.sampler import TrainSampler
from tools.train_se2 import build_model, fwd, collate, evaluate, PCR

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
DATA = '/home/zyf/data/nuscenes'; dev = 'cuda'
out = '/home/zyf/paper/refine-logs/m1_se2trans_Car'; os.makedirs(out, exist_ok=True)
PEAK_LR, WARMUP, HORIZON, MIN_LR, CLIP = 6e-4, 1000, 95000, 1e-4, 1.0
MAX_EPOCHS, MICRO = 30, 128
torch.manual_seed(0); np.random.seed(0)

tcfg = ConfigDict(dict(num_candidates=2, target_thr=None, search_thr=5, point_cloud_range=PCR, time_flip=True, flip=True))
train = TrainSampler(dataset=dict(type='NuScenesDataset', path=DATA, split='train_track',
                                  category_name='Car', preloading=True, preload_offset=10), cfg=tcfg)
val = NuScenesDataset(path=DATA, split='val', category_name='Car', preloading=True)

# --- balanced sampler: 50% static (<0.5m) / 50% in-band (0.5-4.8m), out-crop (>4.8m) excluded ---
ds = train.dataset; start = train.tracklet_start_ids; ncand = train.num_candidates
nframes = ds.get_num_frames_total()
disp = np.zeros(nframes)
for t in range(ds.get_num_tracklets()):
    frames = ds.training_samples[t]
    for fi in range(len(frames)):
        a = start[t] + fi
        if fi == 0:
            disp[a] = 0.0
        else:
            c0 = frames[fi - 1]['3d_bbox'].center; c1 = frames[fi]['3d_bbox'].center
            disp[a] = float(np.linalg.norm(np.array(c1)[:2] - np.array(c0)[:2]))
anno_of_idx = np.arange(len(train)) // ncand
d_idx = disp[anno_of_idx]
is_static = d_idx < 0.5; is_inband = (d_idx >= 0.5) & (d_idx <= 4.8)
n_s, n_i = int(is_static.sum()), int(is_inband.sum())
weights = np.zeros(len(train))
weights[is_static] = 0.5 / max(n_s, 1)
weights[is_inband] = 0.5 / max(n_i, 1)
print(f"balanced sampler: static idx={n_s} in-band idx={n_i} out-crop excluded={int((~is_static&~is_inband).sum())}", flush=True)
sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train), replacement=True)
dl = DataLoader(train, batch_size=MICRO, sampler=sampler, num_workers=6, collate_fn=collate, drop_last=True, persistent_workers=True)
print(f"steps/epoch={len(dl)} val tracklets={val.get_num_tracklets()}", flush=True)

model = build_model('se2trans').to(dev).train()
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-3)
scaler = torch.amp.GradScaler('cuda')


def lr_at(step):
    if step < WARMUP: return PEAK_LR * (step + 1) / WARMUP
    t = min(1.0, (step - WARMUP) / max(1, HORIZON - WARMUP))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * t))


step, t0, best_in, best_epoch, last_ent, last_top1 = 0, time.time(), -1, -1, 0, 0
log = []; nogo = False
for epoch in range(1, MAX_EPOCHS + 1):
    for batch in dl:
        for g in opt.param_groups: g['lr'] = lr_at(step)
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            res = fwd(model, batch['prev'], batch['this'])
            gt = batch['label'].to(dev)
            ld = model['h'].loss(res, {'box_label': list(gt)}); loss = sum(ld.values())
        last_ent, last_top1 = float(res['entropy']), float(res['top1'])
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if torch.isfinite(gn): scaler.step(opt)
        scaler.update(); step += 1
        if step % 250 == 0:
            print(f"  ep{epoch} step{step:6d} lr{lr_at(step):.1e} loss{loss.item():6.3f} ent{last_ent:.2f} top1{last_top1:.3f} {step/(time.time()-t0):.1f}it/s", flush=True)
        if step in (1000, 2000, 3000, 5000, 8000):
            ev = evaluate(model, val, max_tracklets=120)
            ib = ev['inband']['success']
            print(f"[CHECK step{step}] in-band {ib:.2f}(n{ev['inband']['n']}) all {ev['all']['success']:.2f} ent{last_ent:.2f} top1{last_top1:.3f}", flush=True)
            log.append({'step': step, 'inband': ib, 'all': ev['all']['success'], 'ent': last_ent}); json.dump(log, open(f'{out}/attemptB_checks.json', 'w'), indent=2)
            if last_ent > math.log(4096) - 0.3 and step >= 2000:
                print(f"STOP_RULE_NOGO uniform at step{step}", flush=True); nogo = True; break
            if step == 5000 and ib < 14:
                print(f"STOP_RULE_NOGO in-band {ib:.2f}<14 at step5000", flush=True); nogo = True; break
    if nogo: break
    if epoch % 2 == 0:
        ev = evaluate(model, val, max_tracklets=500)
        print(f"[EVAL ep{epoch}] all {ev['all']['success']:.2f}/{ev['all']['precision']:.2f} static {ev['static']['success']:.2f} "
              f"inband {ev['inband']['success']:.2f}/{ev['inband']['precision']:.2f}(n{ev['inband']['n']}) out {ev['outcrop']['success']:.2f}", flush=True)
        log.append({'epoch': epoch, 'step': step, **ev}); json.dump(log, open(f'{out}/eval_log.json', 'w'), indent=2)
        if epoch >= 10 and ev['inband']['success'] > best_in:
            best_in, best_epoch = ev['inband']['success'], epoch; torch.save(model.state_dict(), f'{out}/best.pt')
if nogo:
    print("ATTEMPT_B_NOGO", flush=True)
else:
    print(f"[best ep{best_epoch} inband {best_in:.2f}] full-val eval", flush=True)
    if os.path.exists(f'{out}/best.pt'): model.load_state_dict(torch.load(f'{out}/best.pt'))
    ev = evaluate(model, val, max_tracklets=None)
    print("FINAL:", json.dumps(ev), flush=True)
    json.dump({'final': ev, 'best_epoch': best_epoch, 'log': log}, open(f'{out}/final.json', 'w'), indent=2)
    print("ATTEMPT_B_DONE", flush=True)
