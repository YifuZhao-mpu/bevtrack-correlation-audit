"""M0 step 4: overfit 32 pairs with the ORIGINAL implicit head (dense encoder).
Validates the full harness (encoder + fuser + RLE head + label convention).
Target (Codex): median center err < 0.15 m and yaw err < 2 deg within ~1-2k steps.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmengine.config import ConfigDict
from mmengine.model import revert_sync_batchnorm
from datasets.nuscenes import NuScenesDataset
from datasets.sampler import TrainSampler
from models.voxelnet import VoxelNet
from models.bevfuser import BEVFuser
from models.simple_head import SimpleHead

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DATA = '/home/zyf/data/nuscenes'
PCR = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
NPAIR = 32
STEPS = 1500
dev = 'cuda'

print("loading nuScenes train_track Car (no preload)...")
ds = NuScenesDataset(path=DATA, split='train_track', category_name='Car', preloading=False)
cfg = ConfigDict(dict(num_candidates=2, target_thr=None, search_thr=1,
                      point_cloud_range=PCR, time_flip=False, flip=False))

pairs = []
t = 0
while len(pairs) < NPAIR and t < ds.get_num_tracklets():
    n = ds.get_num_frames_tracklet(t)
    for f in range(n - 1):
        try:
            fr = ds.get_frames(t, [f, f + 1])
            out = TrainSampler.processing({'prev_frame': fr[0], 'this_frame': fr[1], 'candidate_id': 0}, cfg)
            pairs.append(out)
            if len(pairs) >= NPAIR:
                break
        except AssertionError:
            continue
    t += 1
print(f"collected {len(pairs)} fixed pairs")

prev = [p['inputs']['prev_points'].to(dev) for p in pairs]
this = [p['inputs']['this_points'].to(dev) for p in pairs]
gt = torch.stack([p['data_samples']['box_label'] for p in pairs]).to(dev)  # (B,4)

backbone = VoxelNet(PCR, [0.075, 0.075, 0.15], [21, 128, 128])
fuser = BEVFuser()
head = SimpleHead()
model = torch.nn.ModuleDict({'b': backbone, 'f': fuser, 'h': head})
model = revert_sync_batchnorm(model).to(dev).train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda')

print("overfitting...")
t0 = time.time()
for step in range(1, STEPS + 1):
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.float16):
        feats = model['b'](prev + this)          # (2B,128,16,16)
        vec = model['f'](feats)                   # (B,512)
        res = model['h'](vec)
        loss = model['h'].loss(res, {'box_label': list(gt)})['regression_loss']
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    if step % 100 == 0 or step == 1:
        with torch.no_grad():
            pred = res['coors'].float()
            cerr = (pred[:, :2] - gt[:, :2]).norm(dim=1)
            zerr = (pred[:, 2] - gt[:, 2]).abs()
            yerr = (pred[:, 3] - gt[:, 3]).abs()
            yerr = torch.minimum(yerr, 360 - yerr)
            print(f"step {step:4d} loss {float(loss):8.3f} | med center {cerr.median():.3f}m "
                  f"z {zerr.median():.3f}m yaw {yerr.median():.2f}deg")
print(f"elapsed {time.time()-t0:.1f}s")
with torch.no_grad():
    pred = res['coors'].float()
    cerr = (pred[:, :2] - gt[:, :2]).norm(dim=1).median().item()
    yerr = (pred[:, 3] - gt[:, 3]).abs(); yerr = torch.minimum(yerr, 360 - yerr).median().item()
ok = cerr < 0.15 and yerr < 2.0
print(f"FINAL med center {cerr:.3f}m yaw {yerr:.2f}deg -> {'OVERFIT_PASS' if ok else 'OVERFIT_FAIL'}")
