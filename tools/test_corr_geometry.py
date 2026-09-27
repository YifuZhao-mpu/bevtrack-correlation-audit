"""Deterministic geometry tests for the corrected correlation decode
(RESEARCH_REVIEW_2026-07-18, round 2, Q0). Run BEFORE any training:

    python tools/test_corr_geometry.py

Tests (all CPU, no checkpoint — a trained model's accuracy is NOT proof of
coordinate correctness):
  1. One-hot decoder: peak at (8,8)->(0,0); (8,9)->(+0.6,0); (8,7)->(-0.6,0);
     (9,8)->(0,+0.6); (7,8)->(0,-0.6); diagonals and +-7 lags. |err| < 1e-6.
     Plus: flat response decodes to exactly (0,0) (symmetric-support property).
  2. Sub-cell expectation: bilinear probability centered at q+(0.25,-0.40)
     must decode to (+0.15,-0.24) m within 1e-5.
  3. Correlation integration: unique descriptor injected at template (8,8) and
     search (8+ny,8+nx) (zero-filled placement, logit margin >= 30) must decode
     to (0.6*nx, 0.6*ny).
  4. Voxel/axis: translating an asymmetric point pattern by exact multiples of
     8 voxels (0.6 m) moves pillar occupancy by the expected rows/columns with
     matching signs (+x -> +columns, +y -> +rows).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from models.corr_head import BEVCorrFuser, decode_corr_heat, FEATURE_STRIDE_M

H = W = 16
Q = H // 2
FAILURES = []


def check(name, got, want, tol):
    got = torch.as_tensor(got, dtype=torch.float64)
    want = torch.as_tensor(want, dtype=torch.float64)
    err = (got - want).abs().max().item()
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got.tolist()}, want {want.tolist()}, |err|={err:.3e} (tol {tol:g})")
    if not ok:
        FAILURES.append(name)


def test_one_hot_decoder():
    print("[1] one-hot decoder")
    cases = [  # (ky, kx) -> (dx, dy) = ((kx-8)*0.6, (ky-8)*0.6)
        ((8, 8), (0.0, 0.0)),
        ((8, 9), (+0.6, 0.0)),
        ((8, 7), (-0.6, 0.0)),
        ((9, 8), (0.0, +0.6)),
        ((7, 8), (0.0, -0.6)),
        ((9, 9), (+0.6, +0.6)),
        ((7, 7), (-0.6, -0.6)),
        ((15, 15), (+4.2, +4.2)),
        ((1, 1), (-4.2, -4.2)),
        ((1, 15), (+4.2, -4.2)),
        ((15, 1), (-4.2, +4.2)),
    ]
    for (ky, kx), want in cases:
        heat = torch.zeros(1, H, W, dtype=torch.float64)
        heat[0, ky, kx] = 60.0
        disp, conf = decode_corr_heat(heat)
        check(f"peak ({ky},{kx})", disp[0], want, 1e-6)
    disp, _ = decode_corr_heat(torch.zeros(1, H, W, dtype=torch.float64))
    check("flat response", disp[0], (0.0, 0.0), 1e-6)


def test_subcell_expectation():
    print("[2] sub-cell bilinear expectation")
    dx_frac, dy_frac = 0.25, -0.40           # continuous peak at q + (0.25, -0.40)
    p = torch.zeros(H, W, dtype=torch.float64)
    wx = {Q: 1 - dx_frac, Q + 1: dx_frac}                    # x: mass toward col 9
    wy = {Q - 1: -dy_frac, Q: 1 + dy_frac}                   # y: mass toward row 7
    for ky, py in wy.items():
        for kx, px in wx.items():
            p[ky, kx] = py * px
    heat = torch.log(p).unsqueeze(0)                         # softmax(log p) == p
    disp, _ = decode_corr_heat(heat)
    check("bilinear q+(0.25,-0.40)", disp[0],
          (dx_frac * FEATURE_STRIDE_M, dy_frac * FEATURE_STRIDE_M), 1e-5)


def test_correlation_integration():
    print("[3] injected-descriptor correlation integration")
    torch.manual_seed(0)
    fuser = BEVCorrFuser()
    C = 128
    v = torch.full((C,), 2.0)                 # ||v||^2/sqrt(C) = 45.25 -> margin >= 30
    for ny, nx in [(0, 0), (0, 3), (2, 0), (-3, 4), (7, 7), (-7, -7), (-7, 7)]:
        pe = torch.zeros(1, C, H, W)
        te = torch.zeros(1, C, H, W)
        pe[0, :, Q, Q] = v                    # template descriptor at query index
        te[0, :, Q + ny, Q + nx] = v          # zero-filled placement at shifted index
        disp, conf, heat = fuser.correlate(pe, te)
        flat = heat[0].reshape(-1)
        top2 = torch.topk(flat, 2).values
        margin = (top2[0] - top2[1]).item()
        if margin < 30:
            print(f"  FAIL  margin for lag ({ny},{nx}): {margin:.1f} < 30")
            FAILURES.append(f"margin ({ny},{nx})")
        check(f"lag (ny={ny},nx={nx})", disp[0],
              (0.6 * nx, 0.6 * ny), 1e-5)


def test_voxel_axis():
    print("[4] voxel/axis occupancy translation")
    from models.voxelnet import VoxelNet
    net = VoxelNet(point_cloud_range=[-4.8, -4.8, -1.5, 4.8, 4.8, 1.5],
                   voxel_size=[0.075, 0.075, 0.15],
                   grid_size=[21, 128, 128])
    vx = 0.075
    # asymmetric L-shaped pattern, at voxel centers (away from cell boundaries)
    cells = [(60, 60), (61, 60), (62, 60), (60, 61), (60, 62), (63, 65), (70, 58)]
    pts = torch.tensor([[-4.8 + (ix + 0.5) * vx, -4.8 + (iy + 0.5) * vx, 0.0]
                        for ix, iy in cells])

    def occ(points):
        feat = net._pillarize([points])
        return feat[0, 3] > 0                                 # (Hy=rows=y, Wx=cols=x)

    base = occ(pts)
    for (tx, ty), (dc, dr) in [((+0.6, 0.0), (+8, 0)),
                               ((0.0, +0.6), (0, +8)),
                               ((-1.2, +1.8), (-16, +24))]:
        moved = occ(pts + torch.tensor([tx, ty, 0.0]))
        ok = moved.sum() == base.sum()
        ys, xs = torch.nonzero(base, as_tuple=True)
        for iy, ix in zip(ys.tolist(), xs.tolist()):
            ok = ok and bool(moved[iy + dr, ix + dc])
        print(f"  {'PASS' if ok else 'FAIL'}  translate ({tx:+.1f},{ty:+.1f}) m -> "
              f"({dc:+d} cols, {dr:+d} rows)")
        if not ok:
            FAILURES.append(f"voxel shift ({tx},{ty})")


if __name__ == '__main__':
    test_one_hot_decoder()
    test_subcell_expectation()
    test_correlation_integration()
    test_voxel_axis()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("\nALL GEOMETRY TESTS PASS")
