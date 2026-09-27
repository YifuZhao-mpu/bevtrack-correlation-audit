"""Terminal-average protocol statistics from <run>_curve.csv files.

The closed-loop tracking metric fluctuates by 1-2 Success points between
adjacent epochs even once the LR has annealed to ~1e-6, so any single
checkpoint (final or best) is one draw from a wide band. The paper therefore
reports the mean over the last K validation epochs ("terminal average"),
which is selection-free and integrates out the metric's chaotic sensitivity.

Usage: python tools/terminal_stats.py <run> [<run> ...]   (run = csv stem)
Prints, per run: terminal mean+-std for all/static/moving strata, plus raw
best and 9-epoch smoothed peak for reference.
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = 10


def load(stem):
    with open(ROOT / f'{stem}_curve.csv') as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return [float(r[key]) for r in rows if r[key] not in ('', None)]


def smooth(v, k=9):
    return [sum(v[max(0, i - k // 2):min(len(v), i + k // 2 + 1)]) /
            (min(len(v), i + k // 2 + 1) - max(0, i - k // 2)) for i in range(len(v))]


def term(vals):
    t = vals[-K:]
    return statistics.mean(t), (statistics.stdev(t) if len(t) > 1 else 0.0)


def report(stem):
    rows = load(stem)
    eps = [int(r['epoch']) for r in rows]
    out = {'run': stem, 'epochs': eps[-1], 'n_vals': len(rows)}
    for label, skey, pkey in [('all', 'success', 'precision'),
                              ('static', 'static_s', 'static_p'),
                              ('moving', 'inband_s', 'inband_p')]:
        s, p = col(rows, skey), col(rows, pkey)
        if not s:
            continue
        ms, ds = term(s)
        mp, dp = term(p)
        out[label] = (ms, ds, mp, dp)
    s = col(rows, 'success')
    sm = smooth(s)
    b = max(range(len(s)), key=lambda i: s[i])
    sb = max(range(len(sm)), key=lambda i: sm[i])
    out['rawbest'] = (s[b], eps[b])
    out['smpeak'] = (sm[sb], eps[sb])
    return out


if __name__ == '__main__':
    stems = sys.argv[1:] or ['cos500_car', 'cosaug_car', 'corr4c_car',
                             'none800_car', 'disp800_car', 'corr3_car']
    print(f"terminal average = mean over last {K} val epochs")
    for stem in stems:
        try:
            o = report(stem)
        except FileNotFoundError:
            print(f"{stem:16s} -- no curve csv")
            continue
        line = f"{o['run']:16s} ep{o['epochs']:4d}"
        for label in ('all', 'static', 'moving'):
            if label in o:
                ms, ds, mp, dp = o[label]
                line += f"  {label} {ms:5.2f}+-{ds:4.2f}/{mp:5.2f}+-{dp:4.2f}"
        line += f"  | rawbest {o['rawbest'][0]:.2f}@{o['rawbest'][1]}"
        line += f" smpeak {o['smpeak'][0]:.2f}@{o['smpeak'][1]}"
        print(line)
