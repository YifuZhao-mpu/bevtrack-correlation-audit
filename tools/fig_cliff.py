"""Figure: the drift cliff and one-step recovery, from the corrected reruns.

Reads diag_fixed/diag_drift_s*.json and diag_fixed/diag_recovery_s*.json --
the matched arms at their terminal checkpoints, three seeds -- not the
superseded root-level diag_drift.json / diag_recovery.json, which came from a
pre-decode-fix correlation model read at a Precision-selected checkpoint.

Encoded by line style, marker and grey level rather than hue, to match the
other figures in this paper and to survive grayscale printing.
"""
import json
from pathlib import Path

import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf, np.NINF = np.inf, -np.inf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/zyf/paper/BEVTrack')
SEEDS = (0, 1, 2)
INK, MID, LIGHT = '#1a1a1a', '#6e6e6e', '#b4b4b4'

# Validated categorical palette (dataviz reference instance, light mode):
# slot1 blue #2a78d6, slot2 orange #eb6834, slot3 aqua #1baf7a, slot4 yellow #eda100.
# Checked with scripts/validate_palette.js: lightness band, chroma floor, CVD
# separation (worst adjacent dE 9.1 protan) and normal-vision floor all pass; the
# contrast warning is discharged by axis labels, legends and the matching tables.
# Hues are assigned to entities in fixed order and never cycled; every figure also
# carries a non-colour encoding (marker, hatch, line style) so identity never
# depends on colour alone.
C_BASE, C_ADD, C_GATE, C_EVID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'

STRIDE_M = 0.60

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5, 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
})

drift = [json.load(open(ROOT / f'diag_fixed/diag_drift_s{s}.json')) for s in SEEDS]
recov = [json.load(open(ROOT / f'diag_fixed/diag_recovery_s{s}.json')) for s in SEEDS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
offs = [0.0, 0.25, 0.5, 1.0]

# (a) the cliff -- mean over seeds, with the seed range as a band
series = [('baseline', 'static',  ':',  'o', 'implicit, static'),
          ('baseline', 'inband',  '-',  'o', 'implicit, moving'),
          ('corr',     'static',  ':',  's', 'explicit, static'),
          ('corr',     'inband',  '-',  's', 'explicit, moving')]
for arm, stratum, ls, mk, lab in series:
    v = np.array([[d[arm][stratum][str(o)]['mean_err'] for o in offs] for d in drift])
    col = C_BASE if arm == 'baseline' else C_GATE
    ax1.plot(offs, v.mean(0), ls, color=col, marker=mk, ms=3.5, lw=1.4,
             mfc='white' if arm == 'baseline' else col, label=lab)
    ax1.fill_between(offs, v.min(0), v.max(0), color=col, alpha=0.12, lw=0)
ax1.axvline(STRIDE_M, color=LIGHT, ls='--', lw=1)
ax1.text(STRIDE_M - 0.02, 0.30, 'feature stride 0.60 m', rotation=90, fontsize=6.5,
         va='bottom', ha='right', color=MID)
ax1.set_xlabel('injected reference drift $\\delta$ (m)')
ax1.set_ylabel('per-frame $xy$ error (m)')
ax1.grid(color=LIGHT, lw=0.4, alpha=.6)
ax1.set_axisbelow(True)
ax1.legend(fontsize=6.5, loc='upper left', frameon=False)
ax1.set_title('(a) the drift cliff', fontsize=9)
for side in ('top', 'right'):
    ax1.spines[side].set_visible(False)

# (b) recovery -- mean over seeds, seed range as an error bar
labels = ['$\\delta$=0.5\nstatic', '$\\delta$=0.5\nmoving',
          '$\\delta$=1.0\nstatic', '$\\delta$=1.0\nmoving']
cells = [('static', '0.5'), ('inband', '0.5'), ('static', '1.0'), ('inband', '1.0')]
def series_b(arm):
    v = np.array([[r[arm][st][o]['recov_le_0.3'] for st, o in cells] for r in recov])
    return v.mean(0), np.abs(np.vstack([v.mean(0) - v.min(0), v.max(0) - v.mean(0)]))
bm, be = series_b('baseline')
cm, ce = series_b('corr')
x = np.arange(4)
ax2.bar(x - 0.19, bm, 0.38, color=C_BASE, alpha=.30, edgecolor=C_BASE, lw=1.2, yerr=be,
        error_kw=dict(lw=0.8, ecolor=MID), label='implicit')
ax2.bar(x + 0.19, cm, 0.38, color=C_GATE, alpha=.55, edgecolor=C_GATE, lw=1.2, hatch='///', yerr=ce,
        error_kw=dict(lw=0.8, ecolor=MID), label='explicit (gated)')
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
ax2.set_ylabel('recovery rate (%)')
ax2.grid(color=LIGHT, lw=0.4, alpha=.6, axis='y')
ax2.set_axisbelow(True)
ax2.legend(fontsize=6.5, frameon=False)
ax2.set_title('(b) one-step recovery ($\\leq$0.3 m)', fontsize=9)
for side in ('top', 'right'):
    ax2.spines[side].set_visible(False)

fig.tight_layout()
for ext in ('pdf', 'png'):
    out = ROOT / f'paper/pr/figures/drift_cliff.{ext}'
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('wrote', out)
