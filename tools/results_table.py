"""One-glance results table across all runs: best/final overall + final strata.

Sources: <log>_curve.csv (per-epoch, from parse_curves.py) + <tag>.json (last-val strata).
Usage: python tools/results_table.py
"""
import csv
import json
from pathlib import Path

ROOT = Path('/home/zyf/paper/BEVTrack')
# run name -> (curve csv, strat json tag)
RUNS = [
    ('baseline-old(killed160)', 'baseline_car_curve.csv', 'none_strat'),
    ('none800 const-lr', 'none800_car_curve.csv', 'none800'),
    ('disp800 const-lr', 'disp800_car_curve.csv', 'disp800'),
    ('corr v1 additive', 'corr_car_curve.csv', 'corrfair'),
    ('corr v2 gated', 'corr2_car_curve.csv', 'v2fair'),
    ('corr v3 full-glob', 'corr3_car_curve.csv', 'strat'),
    ('cos500 baseline', 'cos500_car_curve.csv', 'cos500'),
    ('cos_aug control', 'cosaug_car_curve.csv', 'cosaug'),
    ('v4a evidence', 'corr4a_car_curve.csv', 'corr4a'),
    ('v4b method', 'corr4b_car_curve.csv', 'corr4b'),
    ('cos500 seed2', 'cos500s2_car_curve.csv', 'cos500s2'),
]


def main():
    print(f"{'run':26s} {'#ep':>4s} {'best(succ)':>16s} {'final':>14s} "
          f"{'fin static':>12s} {'fin inband':>12s}")
    for name, curve, tag in RUNS:
        cp, jp = ROOT / curve, ROOT / f'{tag}.json'
        if not cp.exists():
            continue
        with open(cp) as f:
            rows = [r for r in csv.DictReader(f)]
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r['success']))
        last = rows[-1]
        strata = ''
        st = ib = ''
        if jp.exists():
            j = json.load(open(jp)).get('final', {})
            if 'static' in j:
                st = f"{j['static']['success']:.2f}/{j['static']['precision']:.2f}"
                ib = f"{j['inband']['success']:.2f}/{j['inband']['precision']:.2f}"
        print(f"{name:26s} {last['epoch']:>4s} "
              f"{float(best['success']):6.2f}/{float(best['precision']):5.2f}@{best['epoch']:>3s} "
              f"{float(last['success']):6.2f}/{float(last['precision']):5.2f} "
              f"{st:>12s} {ib:>12s}")


if __name__ == '__main__':
    main()
