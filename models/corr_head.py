"""BEVTrack-Motion: target-conditioned BEV correlation head.

BEVTrack squeezes 3D->BEV, concat(prev,this), then GLOBAL-maxpools -> regresses displacement
from a global vector, discarding the spatial displacement signal. This module correlates the
prev TARGET descriptor (BEV center) against every cell of the this(search) BEV -> a displacement
heatmap -> soft-argmax -> coarse (dx,dy). A global-context branch supplies z/yaw/size + a learned
residual on (dx,dy). No reweighting. Toggle via config (fuser=BEVCorrFuser, head=CorrHead).

Coordinate convention (RESEARCH_REVIEW_2026-07-18 fix): the correlation compares the
query descriptor at integer feature index q = (H//2, W//2) against every search cell k,
so the decoded displacement is query-relative, d = (k - q) * stride, with
stride = 0.075 m voxels x 2^3 stride-2 stages = 0.6 m. Decoding restricts to the
symmetric lag support k in [1, H-1] (lags -(H//2-1)..+(H//2-1)) so a flat softmax
decodes to exactly zero. The pre-fix decode (linspace(-rng, rng, H): cell k at
(k-7.5)*0.64) carried a (+0.32, +0.32) m bias at zero lag that manufactured a
~0.45 m pseudo error floor; every checkpoint trained before 2026-07-18 used it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from .rle_loss import RLELoss
from .bevfuser import ConvReluGN, BEVFuser

FEATURE_STRIDE_M = 0.6  # 0.075 m voxels x 2^3 (three stride-2 stages)


def decode_corr_heat(heat, stride_m=FEATURE_STRIDE_M):
    """Query-relative soft-argmax decode of a correlation response surface.

    heat: (B, H, W) logits, query taken at integer index (H//2, W//2).
    Returns disp (B, 2) [dx, dy] in meters and conf (B, 1) peak probability,
    both computed on the symmetric lag support heat[:, 1:, 1:].
    """
    B, H, W = heat.shape
    qy, qx = H // 2, W // 2
    hs = heat[:, 1:, 1:]                                  # symmetric support: lags -(q-1)..+(q-1)
    prob = F.softmax(hs.reshape(B, -1), dim=1).reshape(B, H - 1, W - 1)
    lags_y = (torch.arange(1, H, device=heat.device, dtype=prob.dtype) - qy) * stride_m
    lags_x = (torch.arange(1, W, device=heat.device, dtype=prob.dtype) - qx) * stride_m
    dy = (prob.sum(2) * lags_y).sum(1)
    dx = (prob.sum(1) * lags_x).sum(1)
    conf = prob.reshape(B, -1).max(1, keepdim=True)[0]    # peak sharpness = correlation confidence
    return torch.stack([dx, dy], 1), conf


@MODELS.register_module()
class BEVCorrFuser(nn.Module):
    def __init__(self, cin=128, rng=4.8, stride_m=FEATURE_STRIDE_M):
        super().__init__()
        self.rng = rng            # crop half-range; kept for config compat (decode no longer uses it)
        self.stride_m = stride_m
        self.embed = nn.Sequential(ConvReluGN(cin, 128), ConvReluGN(128, 128))
        self.glob = BEVFuser()   # FULL none-equivalent 8-conv global branch (v3 ⊇ none)

    def correlate(self, pe, te):
        """(B,C,H,W) embedded prev/this maps -> (disp (B,2), conf (B,1), heat (B,H,W))."""
        B, C, H, W = pe.shape
        tgt = pe[:, :, H // 2, W // 2]                   # (B,C) prev-target descriptor (query index q)
        heat = torch.einsum('bc,bchw->bhw', tgt, te) / (C ** 0.5)
        disp, conf = decode_corr_heat(heat, self.stride_m)
        return disp, conf, heat

    def forward(self, stack_feats):
        prev, this = stack_feats.chunk(2, 0)            # (B,C,H,W) each
        pe, te = self.embed(prev), self.embed(this)     # (B,128,H,W)
        disp, conf, _ = self.correlate(pe, te)
        glob = self.glob(stack_feats)                    # (B,512) full BEVFuser context (= none's backbone)
        return dict(disp=disp, feat=glob, conf=conf)


@MODELS.register_module()
class CorrHead(nn.Module):
    """Correlation (dx,dy) prior + global context -> residual + z/yaw + RLE uncertainty."""

    def __init__(self):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(512, 256, bias=False), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 128, bias=False), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 8))
        self.criterion = RLELoss()

    def forward(self, fused):
        o = self.reg(fused['feat'])                      # (B,8): residual coors[4] + sigma[4]
        coors = o[:, :4].clone()
        coors[:, :2] = coors[:, :2] + fused['disp']      # correlation displacement prior on (dx,dy)
        return {'coors': coors, 'sigma': o[:, 4:]}

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        return {'regression_loss': self.criterion(results['coors'], results['sigma'], gt)}


@MODELS.register_module()
class BEVCorrFuserAttn(nn.Module):
    """v4b fuser (DEFERRED — not in the corrected 2026-07-18 primary arms): attention-
    pooled template instead of the center-cell query. NOTE: the diag_drift.json numbers
    that motivated this design were measured on pre-fix checkpoints trained with the
    biased decode; the "template cliff" mechanism must be re-established before this
    variant re-enters the matrix. Decode here is ABSOLUTE position (template location
    unknown after pooling), cell-center convention: coarse cell k is centered at
    (k - (H/2 - 1/16)) * 0.6 m (input voxel centers at -4.8 + (i+0.5)*0.075)."""

    def __init__(self, cin=128, rng=4.8):
        super().__init__()
        self.rng = rng
        self.embed = nn.Sequential(ConvReluGN(cin, 128), ConvReluGN(128, 128))
        self.tgt_score = nn.Conv2d(128 + 2, 1, 1)
        self.glob = BEVFuser()

    def forward(self, stack_feats):
        prev, this = stack_feats.chunk(2, 0)
        pe, te = self.embed(prev), self.embed(this)
        B, C, H, W = pe.shape
        ys = torch.linspace(-1, 1, H, device=pe.device)
        xs = torch.linspace(-1, 1, W, device=pe.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([gx, gy]).unsqueeze(0).expand(B, -1, -1, -1)
        attn = self.tgt_score(torch.cat([pe, coords], 1)).reshape(B, -1)
        w = F.softmax(attn, 1).reshape(B, 1, H, W)
        tgt = (w * pe).sum((2, 3))                       # (B,C) drift-following template
        heat = torch.einsum('bc,bchw->bhw', tgt, te) / (C ** 0.5)
        prob = F.softmax(heat.reshape(B, -1), dim=1).reshape(B, H, W)
        # Local 3x3 sub-cell centroid around the argmax. (The pre-fix rationale cited
        # a ~0.45 m decode floor that turned out to be the coordinate bug, not a real
        # floor; kept as a decode choice, no longer claimed as a repair.)
        idx = heat.reshape(B, -1).argmax(1)
        y0 = (idx // W).clamp(1, H - 2)                  # window shifts inward at borders
        x0 = (idx % W).clamp(1, W - 2)
        off = torch.tensor([-1, 0, 1], device=pe.device)
        bi = torch.arange(B, device=pe.device)
        patch = heat[bi[:, None, None],
                     (y0[:, None] + off)[:, :, None],
                     (x0[:, None] + off)[:, None, :]]     # (B,3,3)
        pw = F.softmax(patch.reshape(B, 9), 1).reshape(B, 3, 3)
        cell_y = y0.float() + (pw.sum(2) * off.float()).sum(1)
        cell_x = x0.float() + (pw.sum(1) * off.float()).sum(1)
        origin = H / 2 - 0.0625                          # continuous index of x=0 (crop center)
        dy = (cell_y - origin) * FEATURE_STRIDE_M
        dx = (cell_x - origin) * FEATURE_STRIDE_M
        conf = prob.reshape(B, -1).max(1, keepdim=True)[0]
        glob = self.glob(stack_feats)
        return dict(disp=torch.stack([dx, dy], 1), feat=glob, conf=conf)


@MODELS.register_module()
class BEVCorrFuserC2F(nn.Module):
    """v4c: coarse-to-fine correlation (DEFERRED — not in the corrected 2026-07-18
    primary arms). NOTE: the ~0.46 m "feature-stride floor" that motivated this
    design was an artifact of the pre-fix biased decode (corrected 16x16 soft-argmax
    error is ~0.13 m), so the premise for fine-level matching must be re-evaluated.
    Grids below use the cell-center convention: coarse cell k at
    (k - (H/2 - 1/16)) * 0.6 m; fine (64x64, 0.15 m/cell) cell k at (k - 31.5) * 0.15 m.
    Backbone must be built with return_pyramid=True."""

    def __init__(self, cin=128, cin_f=32, rng=4.8, win=7):
        super().__init__()
        self.rng, self.win = rng, win
        # coarse level (16x16): attention-pooled drift-following template
        self.embed = nn.Sequential(ConvReluGN(cin, 128), ConvReluGN(128, 128))
        self.tgt_score = nn.Conv2d(128 + 2, 1, 1)
        # fine level (64x64)
        self.embed_f = nn.Sequential(ConvReluGN(cin_f, 64), ConvReluGN(64, 64))
        self.tgt_score_f = nn.Conv2d(64 + 2, 1, 1)
        self.glob = BEVFuser()

    @staticmethod
    def _coords(B, H, W, device):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx, gy]).unsqueeze(0).expand(B, -1, -1, -1)

    def _attn_template(self, pe, scorer):
        B, C, H, W = pe.shape
        attn = scorer(torch.cat([pe, self._coords(B, H, W, pe.device)], 1)).reshape(B, -1)
        w = F.softmax(attn, 1).reshape(B, 1, H, W)
        return (w * pe).sum((2, 3))                      # (B,C)

    def forward(self, stack_feats):
        f16, f64 = stack_feats['f16'], stack_feats['f64']
        prev, this = f16.chunk(2, 0)
        pe, te = self.embed(prev), self.embed(this)
        B, C, H, W = pe.shape
        tgt = self._attn_template(pe, self.tgt_score)
        heat = torch.einsum('bc,bchw->bhw', tgt, te) / (C ** 0.5)
        prob = F.softmax(heat.reshape(B, -1), 1).reshape(B, H, W)
        grid_c = (torch.arange(H, device=pe.device, dtype=prob.dtype)
                  - (H / 2 - 0.0625)) * FEATURE_STRIDE_M           # absolute cell centers
        disp_c = torch.stack([(prob.sum(1) * grid_c).sum(1),      # soft-argmax: differentiable
                              (prob.sum(2) * grid_c).sum(1)], 1)  # coarse aux target
        conf_c = prob.reshape(B, -1).max(1, keepdim=True)[0]
        idx = heat.reshape(B, -1).argmax(1)                       # window anchor (non-diff)
        py, px = idx // W, idx % W

        # fine level: correlate everywhere, decode only around the coarse peak
        pf, tf = f64.chunk(2, 0)
        pef, tef = self.embed_f(pf), self.embed_f(tf)
        Bf, Cf, Hf, Wf = pef.shape
        tgt_f = self._attn_template(pef, self.tgt_score_f)
        heat_f = torch.einsum('bc,bchw->bhw', tgt_f, tef) / (Cf ** 0.5)
        s = Hf // H                                               # 4: coarse->fine scale
        r = self.win
        cy = (py * s + s // 2).clamp(r, Hf - 1 - r)
        cx = (px * s + s // 2).clamp(r, Wf - 1 - r)
        off = torch.arange(-r, r + 1, device=pe.device)
        bi = torch.arange(B, device=pe.device)
        patch = heat_f[bi[:, None, None],
                       (cy[:, None] + off)[:, :, None],
                       (cx[:, None] + off)[:, None, :]]           # (B,2r+1,2r+1)
        K = 2 * r + 1
        pw = F.softmax(patch.reshape(B, -1), 1).reshape(B, K, K)
        fy = cy.float() + (pw.sum(2) * off.float()).sum(1)
        fx = cx.float() + (pw.sum(1) * off.float()).sum(1)
        fine_stride = 0.15                                        # 0.075 m x 2^1
        disp_f = torch.stack([(fx - (Hf - 1) / 2) * fine_stride,
                              (fy - (Hf - 1) / 2) * fine_stride], 1)
        conf_f = pw.reshape(B, -1).max(1, keepdim=True)[0]

        return dict(disp=disp_f, disp2=disp_c, conf=conf_c, conf2=conf_f,
                    feat=self.glob(f16))


@MODELS.register_module()
class CorrHeadEvidence(nn.Module):
    """v4a/evidence arm: NO forced pathway. Feeds the corr evidence (disp, conf) purely
    as FEATURES: the head may use or ignore it per-sample. Worst case = baseline; any
    gain = value of explicit search evidence. (Pre-fix motivation cited a ~0.45 m
    decode floor / uninformative confidence; both were artifacts of the biased decode.
    Corrected matcher: ~0.13 m soft-argmax error, confidence strongly informative —
    the integration-pathway question is what this arm now measures cleanly.)"""

    def __init__(self, aux_w=0.0, in_extra=3):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(512 + in_extra, 256, bias=False), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 128, bias=False), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 8))
        self.criterion = RLELoss()
        # aux_w > 0: direct L1 on the matcher's disp. Under evidence-only integration
        # the matcher would otherwise train only through the head's use of the feature
        # (v3's matcher quality came from the forced additive path). Logged separately,
        # so matcher learning is observable in the train log.
        self.aux_w = aux_w
        # in_extra: 3 = disp+conf (single-level fusers); 6 = +disp2+conf2 (c2f fuser)
        self.in_extra = in_extra

    def forward(self, fused):
        extras = [fused['disp'], fused['conf']]
        if self.in_extra > 3:
            extras += [fused['disp2'], fused['conf2']]
        o = self.reg(torch.cat([fused['feat']] + extras, 1))
        out = {'coors': o[:, :4], 'sigma': o[:, 4:], 'disp': fused['disp']}
        if 'disp2' in fused:
            out['disp2'] = fused['disp2']
        return out

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        losses = {'regression_loss': self.criterion(results['coors'], results['sigma'], gt)}
        if self.aux_w > 0:
            losses['aux_disp_loss'] = self.aux_w * (results['disp'] - gt[:, :2]).abs().sum(1).mean()
            if 'disp2' in results:   # c2f: coarse level needs its own signal (argmax is non-diff)
                losses['aux_coarse_loss'] = self.aux_w * (results['disp2'] - gt[:, :2]).abs().sum(1).mean()
        return losses


@MODELS.register_module()
class CorrHeadMatched(nn.Module):
    """Information- and capacity-matched integration heads (review 2026-07-25, round 4).

    The three integration pathways are ONE head with one flag, so that comparing
    them changes only the injection rule and nothing else:

        inject='none'     : coors_xy = r                (evidence-only)
        inject='additive' : coors_xy = r + d_corr       (forced)
        inject='gated'    : coors_xy = r + g * d_corr   (gated)

    In every mode the residual `r` is produced by the SAME network from the SAME
    inputs (global feature + d_corr + confidence), which the legacy heads did not
    do: legacy `CorrHead` (forced) saw only the 512-d global feature, so
    evidence-vs-forced confounded the injection pathway with residual input
    information and capacity. Only `gated` additionally instantiates the gate MLP
    (512+1 -> 64 -> 1, ~33k params, 0.2% of the model); param counts are reported.

    The inference-only counterfactual controls (inject_mode / inject_override /
    last_gate / last_disp) are available in all modes for tools/mech_closedloop.py.
    """

    def __init__(self, inject='gated', in_extra=3):
        super().__init__()
        assert inject in ('none', 'additive', 'gated'), inject
        self.inject = inject
        self.in_extra = in_extra
        self.reg = nn.Sequential(
            nn.Linear(512 + in_extra, 256, bias=False), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 128, bias=False), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 8))
        self.gate = (nn.Sequential(nn.Linear(512 + 1, 64), nn.ReLU(True), nn.Linear(64, 1))
                     if inject == 'gated' else None)
        self.criterion = RLELoss()
        self.inject_mode = 'normal'
        self.inject_override = None
        self.last_gate = None
        self.last_disp = None
        # Inference-only control over the EVIDENCE pathway (the d_corr/conf that enter
        # `reg`), mirroring inject_mode/inject_override for the injection pathway.
        # Needed because inject='none' skips the injection branch entirely, so the
        # existing counterfactuals cannot probe whether an evidence-only head is
        # actually using the correlation or merely benefiting from the extra branch.
        # 'shuffled' keeps the marginal distribution of d_corr and destroys only the
        # per-frame correspondence (in-distribution, unlike 'zero'), exactly as
        # shuffled_corr does on the injection path. Defaults leave behaviour unchanged.
        self.evidence_mode = 'normal'
        self.evidence_override = None

    def forward(self, fused):
        disp, feat, conf = fused['disp'], fused['feat'], fused['conf']
        ev = disp
        if self.evidence_mode == 'zero':
            ev = torch.zeros_like(disp)
        elif self.evidence_mode == 'shuffled' and self.evidence_override is not None:
            ev = self.evidence_override.to(disp.device, disp.dtype).reshape(1, 2).expand_as(disp)
        o = self.reg(torch.cat([feat, ev, conf], 1))         # identical in all modes
        self.last_disp = disp.detach()
        coors = o[:, :4].clone()
        if self.inject != 'none':
            g = (torch.sigmoid(self.gate(torch.cat([feat, conf], 1)))
                 if self.inject == 'gated' else torch.ones_like(conf))
            self.last_gate = g.detach()
            inj = disp
            if self.inject_mode == 'a0':
                inj = torch.zeros_like(disp)
            elif self.inject_mode in ('shuffled', 'oracle') and self.inject_override is not None:
                inj = self.inject_override.to(disp.device, disp.dtype).reshape(1, 2).expand_as(disp)
            coors[:, :2] = coors[:, :2] + g * inj
        return {'coors': coors, 'sigma': o[:, 4:], 'disp': disp}

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        return {'regression_loss': self.criterion(results['coors'], results['sigma'], gt)}


@MODELS.register_module()
class CorrHeadGated(nn.Module):
    """v2: learned confidence GATE on the correlation displacement, so a wrong/uncertain
    correlation no longer drags the prediction (the v1 additive-prior flaw). The head also
    sees disp+conf as features and learns when to trust the explicit search vs regression."""

    def __init__(self):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(512 + 3, 256, bias=False), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 128, bias=False), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 8))
        self.gate = nn.Sequential(nn.Linear(512 + 1, 64), nn.ReLU(True), nn.Linear(64, 1))
        self.criterion = RLELoss()
        # inference-only counterfactual controls (mech_pathway closed-loop; default off):
        #   inject_mode: 'normal' | 'a0' (drop injected term) | 'shuffled' | 'oracle'
        #   inject_override: (2,) tensor supplying the shuffled/oracle displacement
        # Untouched during training/normal eval; a checkpoint never stores these.
        self.inject_mode = 'normal'
        self.inject_override = None
        self.last_gate = None            # exposed for logging
        self.last_disp = None            # the matcher's own estimate, for the permutation control

    def forward(self, fused):
        disp, feat, conf = fused['disp'], fused['feat'], fused['conf']
        o = self.reg(torch.cat([feat, disp, conf], 1))   # (B,8) residual + sigma, sees disp+conf
        g = torch.sigmoid(self.gate(torch.cat([feat, conf], 1)))   # (B,1) trust the correlation?
        self.last_gate = g.detach()
        self.last_disp = disp.detach()
        inj = disp
        if self.inject_mode == 'a0':
            inj = torch.zeros_like(disp)
        elif self.inject_mode in ('shuffled', 'oracle') and self.inject_override is not None:
            inj = self.inject_override.to(disp.device, disp.dtype).reshape(1, 2).expand_as(disp)
        coors = o[:, :4].clone()
        coors[:, :2] = coors[:, :2] + g * inj            # GATED displacement prior (mode-controlled)
        return {'coors': coors, 'sigma': o[:, 4:]}

    def loss(self, results, data_samples):
        gt = torch.stack(list(data_samples['box_label'])).to(results['coors'].device)
        return {'regression_loss': self.criterion(results['coors'], results['sigma'], gt)}
