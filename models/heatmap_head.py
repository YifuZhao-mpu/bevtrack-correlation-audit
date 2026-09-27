"""Learned translation HEATMAP head (Codex pivot spec) — reliable alternative to the
fragile fixed-cosine cost volume. search feat (x) broadcast template descriptor ->
4 GN+ReLU conv -> 1ch heatmap -> spatial-softmax soft-argmax for dx,dy; residual MLP
for dz,dyaw only. Tests the same claim: explicit spatial localizer > global-pool regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


def _cbr(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.ReLU(True))


@MODELS.register_module()
class HeatmapHead(nn.Module):
    def __init__(self, in_ch=128, proj_ch=32, up=4, kt=21, pcr=(-4.8, 4.8),
                 sigma_cells=2.0, w_hm=0.2, w_xy=1.0, w_res=0.2, disp_weighted=True,
                 w_static=0.2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_ch, 1, bias=False)
        self.up, self.kt = up, kt
        self.pcr_min, self.pcr_max = pcr
        self.sigma_cells = sigma_cells
        self.w_hm, self.w_xy, self.w_res, self.disp_weighted = w_hm, w_xy, w_res, disp_weighted
        self.w_static = w_static
        self.hm = nn.Sequential(_cbr(3 * proj_ch, 64), _cbr(64, 64), _cbr(64, 32), _cbr(32, 16),
                                nn.Conv2d(16, 1, 1))
        self.res_mlp = nn.Sequential(nn.Linear(proj_ch, 64), nn.ReLU(True), nn.Linear(64, 2))

    def _coords(self, H, device):
        c = (torch.arange(H, device=device).float() + 0.5) / H
        return c * (self.pcr_max - self.pcr_min) + self.pcr_min

    def forward(self, stack_feats, data_samples=None):
        B2 = stack_feats.shape[0]; B = B2 // 2
        tmpl, srch = stack_feats[:B], stack_feats[B:]
        tmpl = self.proj(tmpl); srch = self.proj(srch)
        H = stack_feats.shape[-1] * self.up
        tmpl = F.interpolate(tmpl, size=(H, H), mode='bilinear', align_corners=False)
        srch = F.interpolate(srch, size=(H, H), mode='bilinear', align_corners=False)
        c = H // 2; half = self.kt // 2
        ker = tmpl[:, :, c - half:c + half + 1, c - half:c + half + 1]
        desc = ker.mean(dim=(2, 3))                          # (B,C) descriptor
        descb = desc[:, :, None, None].expand(-1, -1, H, H)
        match = torch.cat([srch, descb, srch * descb], 1)    # (B,3C,H,W)
        logits = self.hm(match).reshape(B, -1)               # (B,H*W)
        prob = logits.softmax(1).reshape(B, H, H)            # (B,H=y,W=x)
        xs = self._coords(H, logits.device)
        px = prob.sum(1); py = prob.sum(2)                   # (B,W) x-marg ; (B,H) y-marg
        dx = (px * xs).sum(1); dy = (py * xs).sum(1)
        res = self.res_mlp(srch.mean(dim=(2, 3)))
        coors = torch.stack([dx, dy, res[:, 0], res[:, 1]], 1)
        with torch.no_grad():
            p = prob.reshape(B, -1)
            ent = -(p * (p + 1e-12).log()).sum(1).mean(); top1 = p.max(1).values.mean()
        return {'coors': coors, 'logits': logits.reshape(B, 1, H, H),
                'soft': torch.stack([dx, dy, torch.zeros_like(dx)], 1),
                'H': H, 'nth': 1, 'entropy': ent, 'top1': top1}

    def _soft_target(self, gt, H, device):
        xs = self._coords(H, device); res_m = (self.pcr_max - self.pcr_min) / H
        sx = self.sigma_cells * res_m
        gx = torch.exp(-((xs[None, :] - gt[:, 0:1]) ** 2) / (2 * sx * sx))
        gy = torch.exp(-((xs[None, :] - gt[:, 1:2]) ** 2) / (2 * sx * sx))
        t = (gy[:, :, None] * gx[:, None, :]).reshape(gt.shape[0], -1)
        return (t / t.sum(1, keepdim=True).clamp(min=1e-9)).reshape(gt.shape[0], H, H)

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        soft = results['soft']; disp = gt[:, :2].norm(dim=1)
        w = torch.where(disp < 0.5, self.w_static, torch.where(disp <= 4.8, 1.0, 0.0)) if self.disp_weighted else torch.ones_like(disp)
        l_xy = (w * F.smooth_l1_loss(soft[:, :2], gt[:, :2], reduction='none').sum(1)).sum() / w.sum().clamp(min=1)
        l_res = F.smooth_l1_loss(results['coors'][:, [2, 3]], gt[:, [2, 3]])
        B = gt.shape[0]
        logp = F.log_softmax(results['logits'].reshape(B, -1), dim=1)
        tgt = self._soft_target(gt, results['H'], gt.device).reshape(B, -1)
        l_hm = (-(tgt * logp).sum(1) * w).sum() / w.sum().clamp(min=1)
        return {'xy': self.w_xy * l_xy, 'hm': self.w_hm * l_hm, 'res': self.w_res * l_res}
