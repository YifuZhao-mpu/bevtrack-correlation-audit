"""M0 step 5: overfit 32 pairs with the SE(2) cost-volume head (dense encoder).
Validates the SE(2) head + volume-CE loss + decode conventions. If signs are wrong
the volume peak won't land -> this is where we calibrate (dx<->W, dy<->H, theta dir).
Target: median center err < 0.3 m, yaw err < 4 deg within ~2k steps (looser than
implicit since the volume is discretized + residual-refined)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmengine.config import ConfigDict
from mmengine.model import revert_sync_batchnorm
from datasets.nuscenes import NuScenesDataset
from datasets.sampler import TrainSampler
from models.voxelnet import VoxelNet
from models.se2_head import SE2Head

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DATA = '/home/zyf/data/nuscenes'
PCR = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
NPAIR, STEPS, dev = 32, 2000, 'cuda'

ds = NuScenesDataset(path=DATA, split='train_track', category_name='Car', preloading=False)
cfg = ConfigDict(dict(num_candidates=2, target_thr=None, search_thr=1,
                      point_cloud_range=PCR, time_flip=False, flip=False))
pairs, t = [], 0
while len(pairs) < NPAIR and t < ds.get_num_tracklets():
    for f in range(ds.get_num_frames_tracklet(t) - 1):
        try:
            fr = ds.get_frames(t, [f, f + 1])
            pairs.append(TrainSampler.processing(
                {'prev_frame': fr[0], 'this_frame': fr[1], 'candidate_id': 0}, cfg))
            if len(pairs) >= NPAIR:
                break
        except AssertionError:
            continue
    t += 1
print(f"collected {len(pairs)} pairs")
prev = [p['inputs']['prev_points'].to(dev) for p in pairs]
this = [p['inputs']['this_points'].to(dev) for p in pairs]
gt = torch.stack([p['data_samples']['box_label'] for p in pairs]).to(dev)
print(f"gt motion ranges: dx[{gt[:,0].min():.2f},{gt[:,0].max():.2f}] dy[{gt[:,1].min():.2f},{gt[:,1].max():.2f}] "
      f"dyaw[{gt[:,3].min():.1f},{gt[:,3].max():.1f}]")

backbone = VoxelNet(PCR, [0.075, 0.075, 0.15], [21, 128, 128])
head = SE2Head(in_ch=128, proj_ch=32, n_theta=15, theta_max=28.0, up=4, kt=21, pcr=(-4.8, 4.8))
model = torch.nn.ModuleDict({'b': backbone, 'h': head})
model = revert_sync_batchnorm(model).to(dev).train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda')

t0 = time.time()
for step in range(1, STEPS + 1):
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.float16):
        feats = model['b'](prev + this)
        res = model['h'](feats)
        losses = model['h'].loss(res, {'box_label': list(gt)})
        loss = sum(losses.values())
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    if step % 100 == 0 or step == 1:
        with torch.no_grad():
            pred = res['coors'].float()
            cerr = (pred[:, :2] - gt[:, :2]).norm(dim=1).median()
            yerr = (pred[:, 3] - gt[:, 3]).abs(); yerr = torch.minimum(yerr, 360 - yerr).median()
            ld = {k: float(v) for k, v in losses.items()}
            print(f"step {step:4d} loss {float(loss):7.3f} {ld} | med center {cerr:.3f}m yaw {yerr:.2f}deg")
print(f"elapsed {time.time()-t0:.1f}s")
with torch.no_grad():
    pred = res['coors'].float()
    cerr = (pred[:, :2] - gt[:, :2]).norm(dim=1).median().item()
    yerr = (pred[:, 3] - gt[:, 3]).abs(); yerr = torch.minimum(yerr, 360 - yerr).median().item()
ok = cerr < 0.3 and yerr < 4.0
print(f"FINAL med center {cerr:.3f}m yaw {yerr:.2f}deg -> {'SE2_OVERFIT_PASS' if ok else 'SE2_OVERFIT_FAIL'}")
