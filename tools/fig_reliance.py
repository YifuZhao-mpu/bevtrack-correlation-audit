"""Figure: reliance on the correlation branch vs gain from it, per trained model.

Both axes are Success differences, but they are measured differently and that
is the point. Gain compares two separately trained networks and so carries the
full training variance. Reliance compares one network against itself under an
inference-time substitution and carries none. The figure shows that the two are
unrelated: models that route a great deal through the branch gain nothing, and
the one model that ignores it entirely performs the same as the rest.
"""
import json
import os
import sys

import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf, np.NINF = np.inf, -np.inf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from paired_bootstrap import tracklet_hist, auc

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

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
})


def S(path, limit=None):
    t = json.load(open(os.path.join(HERE, path)))['tracklets']
    if limit:
        t = t[:limit]
    i, o = np.arange(len(t)), np.ones(len(t))
    g, l, n = tracklet_hist(t, None)
    return auc(g, l, n, i, o)[0]


def main():
    pts = []
    for s in range(3):
        b = S(f'test_evals/m_base_s{s}.json')
        n = S(f'mech_ev/kitti_evid_s{s}_normal.json')
        sh = S(f'mech_ev/kitti_evid_s{s}_shuffled.json')
        pts.append(('KITTI', s, sh - n, n - b))
    for s in range(5):
        b = S(f'test_evals/nus_base_s{s}.json', limit=300)
        n = S(f'mech_ev/nus_evid_s{s}_normal.json')
        sh = S(f'mech_ev/nus_evid_s{s}_shuffled.json')
        pts.append(('nuScenes', s, sh - n, n - b))

    rel = np.array([p[2] for p in pts])
    gain = np.array([p[3] for p in pts])
    r = np.corrcoef(rel, gain)[0, 1]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhline(0, color=LIGHT, lw=0.8, zorder=0)
    ax.axvline(0, color=LIGHT, lw=0.8, zorder=0)
    for ds, s, x, y in pts:
        k = ds == 'KITTI'
        ax.plot([x], [y], 'o' if k else 's', ms=6.5, zorder=3,
                mfc=C_BASE if k else C_ADD, mec=INK, mew=0.8)

    ax.set_xlabel('Reliance: $\\Delta$ Success when the estimate is shuffled')
    ax.set_ylabel('Gain over implicit baseline')
    ax.set_xlim(-17.5, 6.2)
    ax.set_ylim(-5.4, 5.4)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(color=LIGHT, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # the two KITTI models that make the point on their own
    lo = min(pts, key=lambda p: p[2])          # heaviest reliance
    ig = max(pts, key=lambda p: p[2])          # ignores the branch
    ax.annotate('relies most,\ngains nothing', xy=(lo[2], lo[3]), xytext=(lo[2] + 1.0, lo[3] + 2.6),
                fontsize=7.5, color=INK, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', color=MID, lw=0.7, shrinkA=1, shrinkB=4))
    ax.annotate('ignores the branch,\nperforms the same', xy=(ig[2], ig[3]), xytext=(1.5, ig[3] - 1.9),
                fontsize=7.5, color=INK, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=MID, lw=0.7, shrinkA=1, shrinkB=4))
    ax.text(0.03, 0.05, f'$r = {r:+.2f}$  ($n = {len(pts)}$)', transform=ax.transAxes,
            fontsize=8.5, color=INK, ha='left', va='bottom')

    ax.legend(handles=[
        Line2D([], [], marker='o', ls='none', mfc=C_BASE, mec=INK, mew=0.8, ms=6.5, label='KITTI'),
        Line2D([], [], marker='s', ls='none', mfc=C_ADD, mec=INK, mew=0.8, ms=6.5, label='nuScenes'),
    ], loc='upper left', frameon=False, fontsize=8, handletextpad=0.4)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = os.path.join(HERE, 'paper', 'pr', 'figures', f'reliance.{ext}')
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print('wrote', out)
    print(f'r = {r:+.4f}  n = {len(pts)}')


if __name__ == '__main__':
    main()
