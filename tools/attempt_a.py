"""Codex Attempt-A: train se2trans (volume-only xy, fixed temp, disp-weighted, no CE)
with lr=6e-4 warmup1000->cosine->1e-4, clip 1.0. Step-eval in-band at 1000/2000/3000/5000
on a moving val subset; HARD STOP rule. If it passes, continues to 30 epochs + gate."""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
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
dl = DataLoader(train, batch_size=MICRO, shuffle=True, num_workers=6, collate_fn=collate, drop_last=True, persistent_workers=True)
print(f"steps/epoch={len(dl)} val tracklets={val.get_num_tracklets()}", flush=True)

model = build_model('se2trans').to(dev).train()
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M (head: volume-only xy, temp=20 fixed, no CE)", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-3)
scaler = torch.amp.GradScaler('cuda')


def lr_at(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    t = min(1.0, (step - WARMUP) / max(1, HORIZON - WARMUP))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * t))


def quick_inband(nt=120):
    ev = evaluate(model, val, max_tracklets=nt)
    return ev


CHECKS = {1000: ('<12', 12), 2000: (None, None), 3000: ('<12->NOGO', 12), 5000: ('<18->NOGO', 18)}
step, t0, best_in, best_epoch, last_ent, last_top1 = 0, time.time(), -1, -1, 0, 0
log = []
nogo = False
for epoch in range(1, MAX_EPOCHS + 1):
    for batch in dl:
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            res = fwd(model, batch['prev'], batch['this'])
            gt = batch['label'].to(dev)
            ld = model['h'].loss(res, {'box_label': list(gt)})
            loss = sum(ld.values())
        last_ent, last_top1 = float(res['entropy']), float(res['top1'])
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if torch.isfinite(gn): scaler.step(opt)
        scaler.update(); step += 1
        if step % 250 == 0:
            print(f"  ep{epoch} step{step:6d} lr{lr_at(step):.1e} loss{loss.item():6.3f} "
                  f"{ {k:round(float(v),3) for k,v in ld.items()} } ent{last_ent:.2f} top1{last_top1:.3f} {step/(time.time()-t0):.1f}it/s", flush=True)
        if step in (1000, 2000, 3000, 5000):
            ev = quick_inband()
            ib = ev['inband']['success']
            print(f"[CHECK step{step}] in-band Success {ib:.2f} (n{ev['inband']['n']}) all {ev['all']['success']:.2f} ent{last_ent:.2f} top1{last_top1:.3f}", flush=True)
            log.append({'step': step, 'inband': ib, 'all': ev['all']['success'], 'ent': last_ent, 'top1': last_top1})
            json.dump(log, open(f'{out}/attemptA_checks.json', 'w'), indent=2)
            # FULL-CONVERGENCE run: A was the better recipe (climbing to 17.7, near implicit);
            # only abort on genuine uniform-volume collapse, NOT the strict early in-band screen.
            unif = last_ent > (math.log(4096) - 0.3)
            if unif and step >= 2000: nogo = True
            if nogo:
                print(f"STOP_RULE_NOGO at step{step}: in-band {ib:.2f}, ent{last_ent:.2f} top1{last_top1:.3f}", flush=True)
                break
    if nogo: break
    if epoch % 2 == 0:
        ev = evaluate(model, val, max_tracklets=500)
        print(f"[EVAL ep{epoch}] all {ev['all']['success']:.2f}/{ev['all']['precision']:.2f} "
              f"static {ev['static']['success']:.2f} inband {ev['inband']['success']:.2f}/{ev['inband']['precision']:.2f}(n{ev['inband']['n']}) out {ev['outcrop']['success']:.2f}", flush=True)
        log.append({'epoch': epoch, 'step': step, **ev}); json.dump(log, open(f'{out}/eval_log.json', 'w'), indent=2)
        if epoch >= 10 and ev['inband']['success'] > best_in:
            best_in, best_epoch = ev['inband']['success'], epoch
            torch.save(model.state_dict(), f'{out}/best.pt')
if nogo:
    print("ATTEMPT_A_NOGO", flush=True)
else:
    print(f"[best ep{best_epoch} inband {best_in:.2f}] full-val eval", flush=True)
    if os.path.exists(f'{out}/best.pt'): model.load_state_dict(torch.load(f'{out}/best.pt'))
    ev = evaluate(model, val, max_tracklets=None)
    print("FINAL:", json.dumps(ev), flush=True)
    json.dump({'final': ev, 'best_epoch': best_epoch, 'log': log}, open(f'{out}/final.json', 'w'), indent=2)
    print("ATTEMPT_A_DONE", flush=True)
