"""Minimal mean-pillar + 2D-CNN BEV encoder with GroupNorm (Codex-approved pivot).

Replaces the dense-3D-conv surrogate (too slow: ~4 it/s -> can't reach convergence)
and BatchNorm (running-stat eval instability). Per-pillar feature = mean(x,y,z) +
log1p(count) over the z-column; scatter to a 128x128 BEV; 2D CNN (GroupNorm) with the
same stage widths/downsampling -> per-frame BEV [N,128,16,16]. Used IDENTICALLY for
both the implicit and SE(2) heads (encoder is the controlled variable).

Original spconv version at models/voxelnet_spconv_orig.py.bak.
"""
import torch
from torch import nn
from mmengine.registry import MODELS

GN = 8  # groups for GroupNorm (16,32,64,128 all divisible by 8)


def cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(GN, cout),
        nn.ReLU(inplace=True),
    )


@MODELS.register_module()
class VoxelNet(nn.Module):
    """Pillar BEV encoder. forward(list_of_(Ni,3) point tensors) -> (L,128,16,16)."""

    def __init__(self, point_cloud_range, voxel_size, grid_size, return_pyramid=False):
        super().__init__()
        self.register_buffer('pc_range', torch.tensor(point_cloud_range, dtype=torch.float32))
        self.register_buffer('vsize', torch.tensor(voxel_size, dtype=torch.float32))
        self.Hy, self.Wx = int(grid_size[1]), int(grid_size[2])  # 128 x 128
        self.return_pyramid = return_pyramid  # also expose the 64x64 stage (c2f corr)

        self.stem = nn.Sequential(cbr(4, 16), cbr(16, 16))
        self.down1 = cbr(16, 32, stride=2)
        self.conv2 = cbr(32, 32)
        self.down2 = cbr(32, 64, stride=2)
        self.conv3 = cbr(64, 64)
        self.down3 = cbr(64, 128, stride=2)
        self.conv4 = cbr(128, 128)
        self.out = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                 nn.GroupNorm(GN, 128), nn.ReLU(inplace=True))

    @torch.no_grad()
    def _pillarize(self, point_list):
        """Mean-xyz + log1p(count) pillars -> (L,4,Hy,Wx). Vectorized scatter."""
        device = self.pc_range.device
        L = len(point_list)
        Hy, Wx = self.Hy, self.Wx
        x0, y0 = self.pc_range[0], self.pc_range[1]
        vx, vy = self.vsize[0], self.vsize[1]
        cells = Hy * Wx

        pts_all, bid_all = [], []
        for i, pts in enumerate(point_list):
            if pts.numel() == 0:
                continue
            p = pts[:, :3].to(device, dtype=torch.float32)
            pts_all.append(p)
            bid_all.append(torch.full((p.shape[0],), i, device=device, dtype=torch.long))
        if not pts_all:
            return torch.zeros(L, 4, Hy, Wx, device=device)
        P = torch.cat(pts_all, 0)
        B = torch.cat(bid_all, 0)
        ix = ((P[:, 0] - x0) / vx).floor().long()
        iy = ((P[:, 1] - y0) / vy).floor().long()
        m = (ix >= 0) & (ix < Wx) & (iy >= 0) & (iy < Hy)
        ix, iy, P, B = ix[m], iy[m], P[m], B[m]
        flat = B * cells + iy * Wx + ix
        sum_xyz = torch.zeros(L * cells, 3, device=device).index_add_(0, flat, P)
        count = torch.zeros(L * cells, 1, device=device).index_add_(
            0, flat, torch.ones(flat.shape[0], 1, device=device))
        mean = sum_xyz / count.clamp(min=1.0)
        feat = torch.cat([mean, torch.log1p(count)], dim=1)          # (L*cells,4)
        return feat.view(L, Hy, Wx, 4).permute(0, 3, 1, 2).contiguous()

    def forward(self, point_list):
        x = self._pillarize(point_list)   # (L,4,128,128)
        x = self.stem(x)
        f64 = self.conv2(self.down1(x))   # (L,32,64,64) 0.15 m/cell
        x = self.conv3(self.down2(f64))
        x = self.conv4(self.down3(x))     # (L,128,16,16)
        f16 = self.out(x)
        if self.return_pyramid:
            return dict(f16=f16, f64=f64)
        return f16
