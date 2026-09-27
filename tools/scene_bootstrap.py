"""Paired SCENE-cluster bootstrap for eval_tracklets.py outputs.

Why this exists (PR package req#1). tools/paired_bootstrap.py resamples
TRACKLETS, which silently assumes tracklets are independent. They are not:
tracklets from one scene share weather, traffic density, sensor pose and road
geometry, so tracklet-level resampling understates the true uncertainty. The
honest cluster is the SCENE. KITTI Test has only 2 scenes, so that correction
was structurally impossible there -- nuScenes val has 150, which is the whole
reason the nuScenes matrix was run.

Metric definitions, strata and decision rules are IMPORTED from
paired_bootstrap so the numbers stay directly comparable to the KITTI
analysis; the ONLY difference is the resampling unit.

Usage:
  python tools/scene_bootstrap.py armA.json[,armA2,...] armB.json[,armB2,...]
         [--n-boot 10000] [--rope 1.0] [--seed 0]
Comma-joined JSONs are seed-paired exactly as in paired_bootstrap: seed k of A
vs seed k of B share the resampled scene draw, deltas averaged over seeds.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import tracklet_hist, auc  # noqa: E402  (identical metric)


def scene_index(tracklets, label):
    """scene id per tracklet -> (scene_names, per-tracklet scene code)."""
    if not tracklets:
        sys.exit(f'FATAL: {label} has no tracklets')
    missing = [i for i, tr in enumerate(tracklets) if not tr.get('scene')]
    unres = [i for i, tr in enumerate(tracklets) if tr.get('scene') == '?']
    if missing or unres:
        sys.exit(
            f'FATAL: {label}: {len(missing)} tracklets carry no scene field and '
            f'{len(unres)} are unresolved. This JSON predates the scene fix in '
            f'tools/eval_tracklets.py -- re-run the evaluation before any '
            f'scene-level inference.')
    names = sorted({tr['scene'] for tr in tracklets})
    code = {s: k for k, s in enumerate(names)}
    return names, np.array([code[tr['scene']] for tr in tracklets])


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

    names, sc = scene_index(A[0], A_files[0])
    for f, t in list(zip(A_files, A)) + list(zip(B_files, B)):
        n2, sc2 = scene_index(t, f)
        if n2 != names or not np.array_equal(sc2, sc):
            sys.exit(f'FATAL: {f} has a different scene layout than {A_files[0]}; '
                     'arms must be evaluated on an identical frozen split to be paired.')
    ns = len(names)
    # tracklet indices per scene, and per-scene tracklet counts
    members = [np.flatnonzero(sc == k) for k in range(ns)]
    sizes = np.array([len(m) for m in members])
    M = np.zeros((ns, nt))                       # scene -> its tracklets, for weight spread
    for k, m in enumerate(members):
        M[k, m] = 1.0

    print(f'{args.b}  minus  {args.a}')
    print(f'  {nt} tracklets in {ns} scenes, {len(A_files)} seed(s), '
          f'{args.n_boot} boots, resampling unit = SCENE')
    print(f'  tracklets/scene: min {sizes.min()} median {int(np.median(sizes))} '
          f'max {sizes.max()}')
    if ns < 10:
        print(f'  WARNING: only {ns} scenes -- a scene-level CI is not informative '
              f'here (this is exactly why KITTI Test, with 2 scenes, cannot support '
              f'this analysis). Reporting anyway for completeness.')

    full = np.arange(nt)
    mult1 = np.ones(nt)
    rng = np.random.default_rng(args.seed)
    # pre-draw scene resamples so every stratum uses the SAME bootstrap replicates
    draws = rng.integers(0, ns, size=(args.n_boot, ns))

    for stratum in [None, 'static', 'moving']:
        HA = [tracklet_hist(a, stratum) for a in A]
        HB = [tracklet_hist(b, stratum) for b in B]

        def paired_delta(mult):
            ds = 0.0
            for (ga, la, na), (gb, lb, nb) in zip(HA, HB):
                ds += auc(gb, lb, nb, full, mult)[0] - auc(ga, la, na, full, mult)[0]
            return ds / len(HA)

        ds_pt = paired_delta(mult1)
        dp_pt = np.mean([auc(gb, lb, nb, full, mult1)[1] - auc(ga, la, na, full, mult1)[1]
                         for (ga, la, na), (gb, lb, nb) in zip(HA, HB)])

        boot = np.empty(args.n_boot)
        for k in range(args.n_boot):
            smult = np.bincount(draws[k], minlength=ns).astype(float)
            boot[k] = paired_delta(smult @ M)    # a scene's weight hits all its tracklets
        boot = boot[~np.isnan(boot)]
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
