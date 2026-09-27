"""Extract per-epoch val curves (overall + stratified) from BEVTrack training logs.

Usage: python tools/parse_curves.py <log> [<log> ...]
Writes <log-stem>_curve.csv next to each log with columns:
  epoch,success,precision,static_s,static_p,inband_s,inband_p
Stratified block is printed by datasets/metrics.py right before the final
Epoch(val) line; runs older than the stratified patch just get empty strata cols.
"""
import csv
import re
import sys
from pathlib import Path

VAL = re.compile(r'Epoch\(val\) \[(\d+)\]\[\d+/\d+\]\s+success: ([\d.]+)\s+precision: ([\d.]+)')
STRAT = re.compile(r'^\s+(all|static|inband)\s+Success/Precision ([\d.]+)/([\d.]+)')


def parse(path: Path):
    rows, strata = [], {}
    for line in path.read_text(errors='replace').splitlines():
        m = STRAT.match(line)
        if m:
            strata[m.group(1)] = (float(m.group(2)), float(m.group(3)))
            continue
        m = VAL.search(line)
        if m:
            ep, s, p = int(m.group(1)), float(m.group(2)), float(m.group(3))
            st = strata.get('static', ('', ''))
            ib = strata.get('inband', ('', ''))
            rows.append([ep, s, p, st[0], st[1], ib[0], ib[1]])
            strata = {}
    return rows


def main():
    for arg in sys.argv[1:]:
        path = Path(arg)
        rows = parse(path)
        if not rows:
            print(f'{path}: no val entries')
            continue
        out = path.with_name(path.stem.replace('.log', '') + '_curve.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'success', 'precision',
                        'static_s', 'static_p', 'inband_s', 'inband_p'])
            w.writerows(rows)
        best = max(rows, key=lambda r: r[1])
        last = rows[-1]
        print(f'{path.name}: {len(rows)} vals  best ep{best[0]} {best[1]:.2f}/{best[2]:.2f}'
              f'  last ep{last[0]} {last[1]:.2f}/{last[2]:.2f}  -> {out.name}')


if __name__ == '__main__':
    main()
