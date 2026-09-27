"""Cross-category pooled analysis (journal/PR route).

Van (16 Test tracklets) and Cyclist (8) can never resolve on their own, so the
defensible statistic is not four per-category verdicts but:

  1. per-category paired delta (point estimate + its own wide CI, reported honestly);
  2. a POOLED delta across categories, two weightings:
       - equal-weight  : mean of per-category deltas (meta-analysis style; every
                         category counts once, so Car cannot dominate)
       - frame-weighted: all frames pooled into one Success/Precision curve
                         (the "overall benchmark" view Car dominates)
  3. between-category HETEROGENEITY: pairwise interaction contrasts
     (delta_catA - delta_catB) with CIs. A contrast whose CI excludes zero is
     direct evidence that the effect genuinely differs by category, which is the
     paper's thesis — not merely "significant here, not there".

Bootstrap: tracklets are resampled with replacement WITHIN each category
(stratified), the same resampled indices are used for both arms and every seed
(paired), and per-seed deltas are averaged before pooling.

Usage (arms given as category:baseFiles:challengerFiles, comma-joined seeds):
  python tools/pooled_categories.py \
    car:test_evals/fix_base_s0.json,...:test_evals/fix_gated_s0.json,... \
    ped:...:... [van:...:...] [cyc:...:...] [--n-boot 4000] [--stratum moving]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from paired_bootstrap import IOU_AXIS, DIST_AXIS, tracklet_hist  # noqa: E402


def auc_from_counts(cnt_ge, cnt_le, n, mult):
    tot = (n * mult).sum()
    if tot == 0:
        return np.nan, np.nan
    w = mult[:, None]
    succ = (cnt_ge * w).sum(0) / tot
    prec = (cnt_le * w).sum(0) / tot
    return np.trapezoid(succ, IOU_AXIS) * 100, np.trapezoid(prec, DIST_AXIS) * 100 / 2


class Cat:
    """Loaded histograms for one category: base and challenger, per seed."""

    def __init__(self, name, base_files, chal_files, stratum):
        self.name = name
        b = [tracklet_hist(json.load(open(f))['tracklets'], stratum) for f in base_files]
        c = [tracklet_hist(json.load(open(f))['tracklets'], stratum) for f in chal_files]
        assert len(b) == len(c), f'{name}: seed count mismatch'
        self.base, self.chal = b, c
        self.nt = len(b[0][2])

    def delta(self, mult):
        """Mean over seeds of (challenger - base) Success on the resampled tracklets."""
        ds = []
        for (gb, lb, nb), (gc, lc, nc) in zip(self.base, self.chal):
            sb = auc_from_counts(gb, lb, nb, mult)[0]
            sc = auc_from_counts(gc, lc, nc, mult)[0]
            ds.append(sc - sb)
        return float(np.mean(ds))

    def frames(self, mult):
        """Summed frame counts (for frame-weighted pooling), mean over seeds."""
        return float(np.mean([(n * mult).sum() for _, _, n in self.base]))

    def pooled_curves(self, mult):
        """Frame-weighted numerator/denominator contributions for both arms."""
        out = []
        for arm in (self.base, self.chal):
            ge = np.mean([(g * mult[:, None]).sum(0) for g, _, _ in arm], axis=0)
            le = np.mean([(l * mult[:, None]).sum(0) for _, l, _ in arm], axis=0)
            tot = np.mean([(n * mult).sum() for _, _, n in arm])
            out.append((ge, le, tot))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='+', help='cat:baseFiles:chalFiles (comma-joined seeds)')
    ap.add_argument('--n-boot', type=int, default=4000)
    ap.add_argument('--stratum', default=None, choices=[None, 'static', 'moving'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='challenger - base')
    args = ap.parse_args()

    cats = []
    for spec in args.specs:
        name, bf, cf = spec.split(':')
        cats.append(Cat(name, bf.split(','), cf.split(','), args.stratum))
    rng = np.random.default_rng(args.seed)

    def one_draw(mults):
        per = {c.name: c.delta(mults[c.name]) for c in cats}
        equal = float(np.mean(list(per.values())))
        # frame-weighted: pool frames across categories into one curve per arm
        ge_b = le_b = tot_b = 0.0
        ge_c = le_c = tot_c = 0.0
        for c in cats:
            (gb, lb, tb), (gc, lc, tc) = c.pooled_curves(mults[c.name])
            ge_b, le_b, tot_b = ge_b + gb, le_b + lb, tot_b + tb
            ge_c, le_c, tot_c = ge_c + gc, le_c + lc, tot_c + tc
        sb = np.trapezoid(ge_b / tot_b, IOU_AXIS) * 100
        sc = np.trapezoid(ge_c / tot_c, IOU_AXIS) * 100
        return per, equal, float(sc - sb)

    ones = {c.name: np.ones(c.nt) for c in cats}
    per_pt, equal_pt, framew_pt = one_draw(ones)

    boots_per = {c.name: [] for c in cats}
    boots_equal, boots_framew, boots_contrast = [], [], {}
    names = [c.name for c in cats]
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            boots_contrast[(names[a], names[b])] = []

    for _ in range(args.n_boot):
        mults = {}
        for c in cats:
            idx = rng.integers(0, c.nt, c.nt)
            mults[c.name] = np.bincount(idx, minlength=c.nt).astype(float)
        per, equal, framew = one_draw(mults)
        for k, v in per.items():
            boots_per[k].append(v)
        boots_equal.append(equal); boots_framew.append(framew)
        for (x, y) in boots_contrast:
            boots_contrast[(x, y)].append(per[x] - per[y])

    def ci(arr, lo=2.5, hi=97.5):
        return np.percentile(arr, [lo, hi])

    st = args.stratum or 'all'
    print(f'### {args.label}  |  stratum: {st}  |  {args.n_boot} boots')
    print('\nPer-category paired delta Success (own tracklet-cluster CI):')
    for c in cats:
        lo, hi = ci(boots_per[c.name])
        tag = 'CI excludes 0' if lo > 0 or hi < 0 else 'spans 0'
        print(f'  {c.name:4s} (n={c.nt:3d} tracklets): {per_pt[c.name]:+6.2f}  [95% {lo:+6.2f},{hi:+6.2f}]  {tag}')
    lo, hi = ci(boots_equal)
    print(f'\nPOOLED equal-weight  : {equal_pt:+6.2f}  [95% {lo:+6.2f},{hi:+6.2f}]'
          f'  {"CI excludes 0" if lo > 0 or hi < 0 else "spans 0"}')
    lo, hi = ci(boots_framew)
    print(f'POOLED frame-weighted: {framew_pt:+6.2f}  [95% {lo:+6.2f},{hi:+6.2f}]'
          f'  {"CI excludes 0" if lo > 0 or hi < 0 else "spans 0"}')
    print('\nBetween-category heterogeneity (interaction contrasts):')
    for (x, y), arr in boots_contrast.items():
        lo, hi = ci(arr)
        d = per_pt[x] - per_pt[y]
        tag = 'GENUINE difference (CI excludes 0)' if lo > 0 or hi < 0 else 'not resolvable'
        print(f'  {x} - {y}: {d:+6.2f}  [95% {lo:+6.2f},{hi:+6.2f}]  {tag}')


if __name__ == '__main__':
    main()
