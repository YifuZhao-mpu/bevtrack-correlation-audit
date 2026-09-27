"""Diagnostic: overfit 32 pairs with the EXACT M1 se2trans config (pillar + translation_only).
If it overfits -> head works, full-data issue is dynamics (lr/loss). If not -> bug in the path."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from mmengine.config import ConfigDict
from datasets.nuscenes import NuScenesDataset
from datasets.sampler import TrainSampler
from tools.train_se2 import build_model, fwd, PCR

dev = 'cuda'
ds = NuScenesDataset(path='/home/zyf/data/nuscenes', split='train_track', category_name='Car', preloading=False)
cfg = ConfigDict(dict(num_candidates=2, target_thr=None, search_thr=1, point_cloud_range=PCR, time_flip=False, flip=False))
pairs, t = [], 0
while len(pairs) < 32 and t < ds.get_num_tracklets():
    for f in range(ds.get_num_frames_tracklet(t) - 1):
        try:
            fr = ds.get_frames(t, [f, f + 1])
            pairs.append(TrainSampler.processing({'prev_frame': fr[0], 'this_frame': fr[1], 'candidate_id': 0}, cfg))
            if len(pairs) >= 32: break
        except AssertionError: continue
    t += 1
prev = [p['inputs']['prev_points'] for p in pairs]
this = [p['inputs']['this_points'] for p in pairs]
gt = torch.stack([p['data_samples']['box_label'] for p in pairs]).to(dev)
print(f"gt dx[{gt[:,0].min():.2f},{gt[:,0].max():.2f}] dy[{gt[:,1].min():.2f},{gt[:,1].max():.2f}] dyaw[{gt[:,3].min():.1f},{gt[:,3].max():.1f}]")

model = build_model('se2trans').to(dev).train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # higher lr for overfit
scaler = torch.amp.GradScaler('cuda')
for step in range(1, 1501):
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.float16):
        res = fwd(model, prev, this)
        ld = model['h'].loss(res, {'box_label': list(gt)})
        loss = sum(ld.values())
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if step % 200 == 0 or step == 1:
        with torch.no_grad():
            pred = res['coors'].float()
            cerr = (pred[:, :2] - gt[:, :2]).norm(dim=1).median()
            yerr = (pred[:, 3] - gt[:, 3]).abs(); yerr = torch.minimum(yerr, 360 - yerr).median()
            lc = {k: round(float(v), 3) for k, v in ld.items()}
            print(f"step{step:4d} loss{float(loss):6.3f} {lc} center{cerr:.3f}m yaw{yerr:.2f}")
# sample preds vs gt
with torch.no_grad():
    pred = res['coors'].float()
    print("pred[:4] dx,dy,dz,dyaw:", np.round(pred[:4].cpu().numpy(), 3).tolist())
    print("gt  [:4] dx,dy,dz,dyaw:", np.round(gt[:4].cpu().numpy(), 3).tolist())
    print("soft[:4] dx,dy,dth   :", np.round(res['soft'][:4].float().cpu().numpy(), 3).tolist())
