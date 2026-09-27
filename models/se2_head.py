"""SE(2)/translation cost-volume localizer head for LiDAR 3D SOT.

Codex Attempt-A mode (xy_from_volume_only + fixed temp + displacement-weighted xy
loss, no CE): forces dx,dy to come ONLY from the volume soft-argmax (no residual
shortcut), so the volume MUST become discriminative for displaced targets; residual
MLP predicts only dz,dyaw. Pairs with IdentityFuser.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


@MODELS.register_module()
class IdentityFuser(nn.Module):
    def forward(self, x):
        return x


@MODELS.register_module()
class SE2Head(nn.Module):
    def __init__(self, in_ch=128, proj_ch=32, n_theta=15, theta_max=28.0,
                 up=4, kt=21, pcr=(-4.8, 4.8), temp=20.0, learnable_temp=False,
                 xy_from_volume_only=True, disp_weighted=True,
                 w_xy=1.0, w_res=0.2, w_vol=0.2, sigma_cells=2.0, translation_only=True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_ch, 1, bias=False)
        self.n_theta = n_theta
        self.kt = kt
        self.up = up
        self.pcr_min, self.pcr_max = pcr
        self.translation_only = translation_only
        self.xy_from_volume_only = xy_from_volume_only
        self.disp_weighted = disp_weighted
        self.w_xy, self.w_res, self.w_vol = w_xy, w_res, w_vol
        self.sigma_cells = sigma_cells
        thetas = torch.linspace(-theta_max, theta_max, n_theta) if n_theta > 1 else torch.zeros(1)
        self.register_buffer('thetas', thetas)
        if learnable_temp:
            self.temp = nn.Parameter(torch.tensor(float(temp)))
        else:
            self.register_buffer('temp', torch.tensor(float(temp)))
        # residual MLP: 2 outputs (dz,dyaw) if xy from volume only, else 4
        n_res = 2 if xy_from_volume_only else 4
        self.res_mlp = nn.Sequential(nn.Linear(proj_ch, 64), nn.ReLU(True), nn.Linear(64, n_res))

    def _rotate_kernel(self, k, deg):
        if deg == 0.0:
            return k
        B = k.shape[0]
        rad = math.radians(deg)
        cos, sin = math.cos(rad), math.sin(rad)
        theta = k.new_tensor([[cos, -sin, 0.0], [sin, cos, 0.0]]).unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(theta, k.shape, align_corners=False)
        return F.grid_sample(k, grid, align_corners=False, padding_mode='zeros')

    def _metric_coords(self, H, device):
        c = (torch.arange(H, device=device).float() + 0.5) / H
        return c * (self.pcr_max - self.pcr_min) + self.pcr_min

    def forward(self, stack_feats, data_samples=None):
        B2 = stack_feats.shape[0]
        B = B2 // 2
        tmpl, srch = stack_feats[:B], stack_feats[B:]
        tmpl = F.normalize(self.proj(tmpl), dim=1)
        srch = F.normalize(self.proj(srch), dim=1)
        H = stack_feats.shape[-1] * self.up
        tmpl = F.normalize(F.interpolate(tmpl, size=(H, H), mode='bilinear', align_corners=False), dim=1)
        srch = F.normalize(F.interpolate(srch, size=(H, H), mode='bilinear', align_corners=False), dim=1)

        c = H // 2
        half = self.kt // 2
        ker = tmpl[:, :, c - half:c + half + 1, c - half:c + half + 1].contiguous()
        C = ker.shape[1]
        srch_g = srch.reshape(1, B * C, H, H)
        thetas = [0.0] if self.translation_only else self.thetas.tolist()
        vols = []
        for th in thetas:
            rk = self._rotate_kernel(ker, th)
            out = F.conv2d(srch_g, rk, padding=half, groups=B)
            vols.append(out.reshape(B, 1, H, H))
        V = torch.cat(vols, 1) / (self.kt * self.kt)
        nth = V.shape[1]
        logits = (V * self.temp).reshape(B, -1)
        prob = logits.softmax(1).reshape(B, nth, H, H)
        xs = self._metric_coords(H, V.device)
        ths = (torch.zeros(1, device=V.device) if self.translation_only else self.thetas)
        px = prob.sum(dim=(1, 2)); py = prob.sum(dim=(1, 3)); pth = prob.sum(dim=(2, 3))
        dx = (px * xs).sum(1); dy = (py * xs).sum(1); dth = (pth * ths).sum(1)

        pooled = srch.mean(dim=(2, 3))
        res = self.res_mlp(pooled)
        if self.xy_from_volume_only:
            dz, dyaw_res = res[:, 0], res[:, 1]
            coors = torch.stack([dx, dy, dz, dth + dyaw_res], dim=1)
        else:
            coors = torch.stack([dx + res[:, 0], dy + res[:, 1], res[:, 2], dth + res[:, 3]], dim=1)
        with torch.no_grad():
            p = prob.reshape(B, -1)
            entropy = -(p * (p + 1e-12).log()).sum(1).mean()
            top1 = p.max(1).values.mean()
        return {'coors': coors, 'V': V, 'logits': logits.reshape(B, nth, H, H),
                'soft': torch.stack([dx, dy, dth], 1), 'H': H, 'nth': nth,
                'entropy': entropy, 'top1': top1}

    def _soft_target(self, gt, H, nth, device):
        B = gt.shape[0]
        xs = self._metric_coords(H, device)
        res_m = (self.pcr_max - self.pcr_min) / H
        sx = self.sigma_cells * res_m
        gx = torch.exp(-((xs[None, :] - gt[:, 0:1]) ** 2) / (2 * sx * sx))
        gy = torch.exp(-((xs[None, :] - gt[:, 1:2]) ** 2) / (2 * sx * sx))
        t = (gy[:, :, None] * gx[:, None, :])[:, None]
        t = t.reshape(B, -1); t = t / t.sum(1, keepdim=True).clamp(min=1e-9)
        return t.reshape(B, nth, H, H)

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        soft = results['soft']
        disp = gt[:, :2].norm(dim=1)
        if self.disp_weighted:
            w = torch.where(disp < 0.5, 0.2, torch.where(disp <= 4.8, 1.0, 0.0))
        else:
            w = torch.ones_like(disp)
        xy_err = F.smooth_l1_loss(soft[:, :2], gt[:, :2], reduction='none').sum(1)
        l_xy = (w * xy_err).sum() / w.sum().clamp(min=1.0)
        l_res = F.smooth_l1_loss(results['coors'][:, [2, 3]], gt[:, [2, 3]])
        out = {'xy': self.w_xy * l_xy, 'res': self.w_res * l_res}
        if self.w_vol > 0:
            B = gt.shape[0]
            logp = F.log_softmax(results['logits'].reshape(B, -1), dim=1)
            tgt = self._soft_target(gt, results['H'], results['nth'], gt.device).reshape(B, -1)
            out['vol'] = self.w_vol * (-(tgt * logp).sum(1) * w).sum() / w.sum().clamp(min=1.0)
        return out
