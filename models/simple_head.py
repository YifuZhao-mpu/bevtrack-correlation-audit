import torch
import torch.nn as nn
from .rle_loss import RLELoss
from mmengine.registry import MODELS


@MODELS.register_module()
class SimpleHead(nn.Module):
    """BEVTrack implicit regression head (LayerNorm, was SyncBatchNorm)."""

    def __init__(self, disp_weighted=False, w_static=0.2, weight_scheme='none', focal_gamma=2.0):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256, bias=False), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 128, bias=False), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 8))
        self.criterion = RLELoss()
        # back-compat: disp_weighted=True maps to scheme='disp' unless an explicit scheme is set
        if disp_weighted and weight_scheme == 'none':
            weight_scheme = 'disp'
        self.weight_scheme, self.w_static, self.focal_gamma = weight_scheme, w_static, focal_gamma

    def forward(self, feats):
        res = self.regression_head(feats)
        return {'coors': res[:, :4], 'sigma': res[:, 4:]}

    def _weights(self, coors, gt):
        """Per-sample loss weights. All schemes EXCLUDE out-of-crop (>4.8m, impossible
        for both heads) for a fair comparison against the disp scheme. Returns None (=uniform)."""
        s = self.weight_scheme
        if s == 'none':
            return None
        disp = gt[:, :2].norm(dim=1)
        oob = disp > 4.8                                   # out-of-crop, excluded everywhere
        if s == 'disp':                                    # semantic displacement strata (the method)
            w = torch.where(disp < 0.5, self.w_static, torch.ones_like(disp))
        elif s == 'invfreq':                               # data-driven inverse stratum frequency
            stat = disp < 0.5
            n_s = stat.sum().clamp(min=1).float(); n_m = (~stat & ~oob).sum().clamp(min=1).float()
            w = torch.where(stat, 1.0/n_s, 1.0/n_m)
        elif s in ('harderr', 'focal'):                    # generic difficulty (detached error magnitude)
            with torch.no_grad():
                err = (coors[:, :2] - gt[:, :2]).norm(dim=1)
            if s == 'harderr':
                w = err / (err[~oob].mean().clamp(min=1e-6))
            else:                                          # focal-style: down-weight easy samples
                w = (err / (err[~oob].max().clamp(min=1e-6))).clamp(0, 1) ** self.focal_gamma
        else:
            raise ValueError(s)
        w = torch.where(oob, torch.zeros_like(w), w)
        return w / w.mean().clamp(min=1e-6)                # normalize mean→1 (comparable loss scale)

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        w = self._weights(results['coors'], gt)
        return {'regression_loss': self.criterion(results['coors'], results['sigma'], gt, weight=w)}
