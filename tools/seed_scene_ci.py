"""Two-level (seed x cluster) bootstrap -- the calibrated replacement for the
single-level CI used in the draft.

Why. tools/paired_bootstrap.py and tools/scene_bootstrap.py resample test items
(tracklets / scenes) but CONDITION ON THE TRAINED MODELS, so training-run
variance is structurally outside their error model. Measured on this project:

  * KITTI Car, identical config / seed / code / environment / eval, two
    independent runs differ by +2.13 and +1.72 Success (all), +3.32 and +2.90
    (static). The pre-registered decision threshold is 1.0.
  * nuScenes, the SAME forced-base contrast on two seeds: moving stratum gives
    +0.64 and -4.98 -- opposite signs -- while the pooled single-level CI
    reports [-2.87, -1.42] and calls it resolved.
  * Two seeds of the SAME implicit baseline are declared a WIN over each other
    on the moving stratum: +3.35 [+2.14, +4.61]. A false positive by
    construction.

The failure gets WORSE as the test set grows: more items shrink the item-level
CI while training noise stays put, so on nuScenes (3660 tracklets, 150 scenes)
the reported interval is several times narrower than the seed-to-seed spread.

What this does. Each replicate resamples SEEDS with replacement and CLUSTERS
with replacement, so both components propagate. Reported next to the
single-level interval and the per-seed deltas, so the inflation is explicit.
Also reports a seed-level t-interval, which makes no cluster assumption at all.

With few seeds the two-level interval is coarse (K seeds give only C(2K-1,K)
distinct multisets) and the t-interval is wide. That is the honest answer: K=2
cannot resolve a training-variance component, and a narrow interval there is an
artefact, not information.

Usage:
  python tools/seed_scene_ci.py A_s0.json,A_s1.json B_s0.json,B_s1.json
         [--unit scene|tracklet] [--n-boot 4000] [--rope 1.0]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import tracklet_hist, auc  # identical metric definition


def cluster_matrix(tracklets, unit, label):
    """(ncl, nt) indicator of which tracklets belong to which resampling cluster."""
    nt = len(tracklets)
    if unit == 'tracklet':
        return np.eye(nt), [str(i) for i in range(nt)]
    miss = [i for i, tr in enumerate(tracklets) if not tr.get('scene') or tr.get('scene') == '?']
    if miss:
        sys.exit(f'FATAL: {label}: {len(miss)} tracklets lack a usable scene id. '
                 'Re-run eval_tracklets.py (its scene fix landed 2026-09-12) or '
                 'pass --unit tracklet.')
    names = sorted({tr['scene'] for tr in tracklets})
    code = {s: k for k, s in enumerate(names)}
    M = np.zeros((len(names), nt))
    for i, tr in enumerate(tracklets):
        M[code[tr['scene']], i] = 1.0
    return M, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--unit', choices=['scene', 'tracklet'], default='scene')
    ap.add_argument('--n-boot', type=int, default=4000)
    ap.add_argument('--rope', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    A_files, B_files = args.a.split(','), args.b.split(',')
    assert len(A_files) == len(B_files), 'arm seed counts differ'
    K = len(A_files)
    A = [json.load(open(f))['tracklets'] for f in A_files]
    B = [json.load(open(f))['tracklets'] for f in B_files]
    nt = len(A[0])
    M, names = cluster_matrix(A[0], args.unit, A_files[0])
    ncl = M.shape[0]

    print(f'{args.b}\n  minus {args.a}')
    print(f'  {nt} tracklets, {ncl} {args.unit}s, {K} seed(s), {args.n_boot} boots')
    if K < 3:
        print(f'  WARNING: K={K} seeds. The training-variance component cannot be '
              f'estimated from {K} runs; the two-level interval below is coarse and '
              f'the t-interval is wide BY CONSTRUCTION. Treat both as lower bounds '
              f'on the true uncertainty, not as a result.')

    full = np.arange(nt); ones = np.ones(nt)
    rng = np.random.default_rng(args.seed)
    seed_draws = rng.integers(0, K, size=(args.n_boot, K))
    cl_draws = rng.integers(0, ncl, size=(args.n_boot, ncl))

    for stratum in [None, 'static', 'moving']:
        HA = [tracklet_hist(a, stratum) for a in A]
        HB = [tracklet_hist(b, stratum) for b in B]

        def delta(k, mult):
            (ga, la, na), (gb, lb, nb) = HA[k], HB[k]
            return auc(gb, lb, nb, full, mult)[0] - auc(ga, la, na, full, mult)[0]

        per_seed = [delta(k, ones) for k in range(K)]
        pt = float(np.mean(per_seed))

        # level 1 only: clusters resampled, models fixed (what the draft reports)
        one = np.empty(args.n_boot)
        for i in range(args.n_boot):
            mult = np.bincount(cl_draws[i], minlength=ncl).astype(float) @ M
            one[i] = np.mean([delta(k, mult) for k in range(K)])
        # level 1 + 2: seeds AND clusters resampled
        two = np.empty(args.n_boot)
        for i in range(args.n_boot):
            mult = np.bincount(cl_draws[i], minlength=ncl).astype(float) @ M
            two[i] = np.mean([delta(k, mult) for k in seed_draws[i]])

        o_lo, o_hi = np.percentile(one, [2.5, 97.5])
        t_lo, t_hi = np.percentile(two, [2.5, 97.5])
        # seed-level t-interval: no cluster model at all
        half = float('inf')
        if K > 1:
            sd = float(np.std(per_seed, ddof=1))
            from math import sqrt
            tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(K, 1.96)
            half = tcrit * sd / sqrt(K)
            tt = f'[{pt - half:+.2f},{pt + half:+.2f}]'
        else:
            tt = 'n/a'

        # The rule is: |d| >= 1.0 AND the two-level interval excludes zero AND the
        # seed-level t interval agrees. Checking only the two-level interval -- which
        # this line did until 2026-09-23 -- labels four comparisons RESOLVED that the
        # rule calls unresolved, because a bootstrap endpoint can sit within Monte
        # Carlo error of zero while the seed-level interval is plainly wide. The
        # variables are renamed here: t_lo/t_hi were the TWO-LEVEL bounds, which is
        # what made the omission easy to miss.
        two_lo, two_hi = t_lo, t_hi
        t_excl = (K > 1) and ((pt - half) * (pt + half) > 0)
        verdict = ('RESOLVED' if (abs(pt) >= 1.0 and two_lo * two_hi > 0 and t_excl)
                   else 'NOT RESOLVED')
        print(f'  {stratum or "all":7s} d={pt:+.2f}  per-seed {["%+.2f" % x for x in per_seed]}')
        print(f'          {args.unit}-only  [{o_lo:+.2f},{o_hi:+.2f}] (width {o_hi-o_lo:.2f})'
              f'   <- conditions on these trained models')
        print(f'          seed+{args.unit} [{t_lo:+.2f},{t_hi:+.2f}] (width {t_hi-t_lo:.2f})'
              f'   -> {verdict}')
        print(f'          seed t-int  {tt}')


if __name__ == '__main__':
    main()
