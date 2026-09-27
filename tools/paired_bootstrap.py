"""Paired tracklet-cluster bootstrap for eval_tracklets.py outputs
(pre-registered decision rules, review 2026-07-18 round 2, Q1).

Aggregate metric mirrors TrackAccuracy exactly: Success = AUC of the pooled
3D-IoU curve (succ[t]=mean(iou>=t), t in linspace(0,1,21)) x100; Precision =
AUC of the pooled center-error curve (prec[t]=mean(dist<=t), t in
linspace(0,2,21)) x100/2, frames pooled over resampled tracklets. Bootstrap
resamples TRACKLETS with replacement, paired across arms. Vectorized via
per-tracklet threshold-count histograms so 10k boots is instant.

Usage:
  python tools/paired_bootstrap.py armA.json[,armA2,...] armB.json[,armB2,...]
         [--n-boot 10000] [--rope 1.0]
Multiple comma-joined JSONs per arm are pooled per-seed-paired: seed k of A vs
seed k of B share bootstrap tracklet indices (paired), deltas averaged over
seeds — the multi-seed paired verdict. Win: mean paired dSuccess >= +1.0 with
95% CI excluding 0. Parity: 90% CI inside +-ROPE.
"""
import argparse
import json

import numpy as np

N = 21
IOU_AXIS = np.linspace(0, 1, N)
DIST_AXIS = np.linspace(0, 2, N)


def tracklet_hist(tracklets, stratum=None, thr=0.5):
    """Per-tracklet threshold-count histograms.

    Returns cnt_ge (T,N) frames with iou>=IOU_AXIS[t], cnt_le (T,N) frames with
    dist<=DIST_AXIS[t], n (T,) frame count — restricted to the stratum.
    """
    cnt_ge, cnt_le, n = [], [], []
    for tr in tracklets:
        iou = np.asarray(tr['iou']); dist = np.asarray(tr['dist']); disp = np.asarray(tr['disp'])
        if stratum == 'static':
            m = disp < thr
        elif stratum == 'moving':
            m = disp >= thr
        else:
            m = np.ones(len(iou), bool)
        iou, dist = iou[m], dist[m]
        cnt_ge.append((iou[:, None] >= IOU_AXIS[None, :]).sum(0))
        cnt_le.append((dist[:, None] <= DIST_AXIS[None, :]).sum(0))
        n.append(len(iou))
    return np.array(cnt_ge), np.array(cnt_le), np.array(n, float)


def auc(cnt_ge, cnt_le, n, idx, mult):
    """Pooled Success/Precision over tracklets idx with multiplicities mult."""
    w = mult[:, None]
    tot = (n[idx] * mult).sum()
    if tot == 0:
        return np.nan, np.nan
    succ = (cnt_ge[idx] * w).sum(0) / tot
    prec = (cnt_le[idx] * w).sum(0) / tot
    return np.trapezoid(succ, IOU_AXIS) * 100, np.trapezoid(prec, DIST_AXIS) * 100 / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--rope', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    A_files = args.a.split(','); B_files = args.b.split(',')
    assert len(A_files) == len(B_files), 'arm seed counts differ'
    A = [json.load(open(f))['tracklets'] for f in A_files]
    B = [json.load(open(f))['tracklets'] for f in B_files]
    nt = len(A[0])
    rng = np.random.default_rng(args.seed)

    label = f'{args.b}  minus  {args.a}'
    print(f'{label}  ({nt} tracklets, {len(A_files)} seed(s), {args.n_boot} boots)')
    for stratum in [None, 'static', 'moving']:
        HA = [tracklet_hist(a, stratum) for a in A]
        HB = [tracklet_hist(b, stratum) for b in B]
        full = np.arange(nt); ones = np.ones(nt)

        def paired_delta(idx):
            mult = np.bincount(idx, minlength=nt).astype(float)
            ds = dp = 0.0
            for (ga, la, na), (gb, lb, nb) in zip(HA, HB):
                sa, pa = auc(ga, la, na, full, mult)
                sb, pb = auc(gb, lb, nb, full, mult)
                ds += sb - sa; dp += pb - pa
            return ds / len(HA), dp / len(HA)

        ds_pt, dp_pt = paired_delta(full)  # point estimate: mult=ones via full+bincount
        # point estimate uses each tracklet once
        mult1 = np.ones(nt)
        ds_pt = np.mean([auc(gb, lb, nb, full, mult1)[0] - auc(ga, la, na, full, mult1)[0]
                         for (ga, la, na), (gb, lb, nb) in zip(HA, HB)])
        dp_pt = np.mean([auc(gb, lb, nb, full, mult1)[1] - auc(ga, la, na, full, mult1)[1]
                         for (ga, la, na), (gb, lb, nb) in zip(HA, HB)])

        boot = np.empty(args.n_boot)
        for k in range(args.n_boot):
            idx = rng.integers(0, nt, nt)
            boot[k] = paired_delta(idx)[0]
        lo95, hi95 = np.percentile(boot, [2.5, 97.5])
        lo90, hi90 = np.percentile(boot, [5, 95])
        verdict = []
        if ds_pt >= 1.0 and lo95 > 0: verdict.append('WIN(B)')
        if ds_pt <= -1.0 and hi95 < 0: verdict.append('WIN(A)')
        if -args.rope < lo90 and hi90 < args.rope: verdict.append('PARITY')
        if not verdict: verdict.append('INCONCLUSIVE')
        print(f'  {stratum or "all":7s} dSucc {ds_pt:+.2f} [95% {lo95:+.2f},{hi95:+.2f}]'
              f' [90% {lo90:+.2f},{hi90:+.2f}] dPrec {dp_pt:+.2f}  -> {" ".join(verdict)}')


if __name__ == '__main__':
    main()
