"""M0 step 0: encoder smoke test on Blackwell sm_120 (Codex-prescribed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.voxelnet import VoxelNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

m = VoxelNet(
    point_cloud_range=[-4.8, -4.8, -1.5, 4.8, 4.8, 1.5],
    voxel_size=[0.075, 0.075, 0.15],
    grid_size=[21, 128, 128],
).cuda().train()
nparam = sum(p.numel() for p in m.parameters())
print(f"VoxelNet params: {nparam/1e6:.2f}M")

lo = torch.tensor([-4.8, -4.8, -1.5], device='cuda')
hi = torch.tensor([4.8, 4.8, 1.5], device='cuda')
pts = [lo + (hi - lo) * torch.rand(4096, 3, device='cuda') for _ in range(2)]

with torch.autocast('cuda', dtype=torch.float16):
    y = m(pts)
    assert y.shape == (2, 128, 16, 16), y.shape
    loss = y.square().mean()
loss.backward()
gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
print(f"OK forward+backward: y={tuple(y.shape)} loss={float(loss):.4f} gradnorm={gnorm:.2f}")

# empty + tiny point set robustness
y2 = m([torch.zeros(1, 3, device='cuda'), lo + (hi-lo)*torch.rand(10, 3, device='cuda')])
assert y2.shape == (2, 128, 16, 16)
print("OK edge cases (empty/tiny point sets)")
print("SMOKE_ENCODER_PASS")
