"""Three data figures, all recomputed from the frozen evaluations.

(1) fig_seeds      -- every run's Success, by arm and dataset. The point is that
                      the spread within an arm is comparable to the spacing
                      between arms, which a table of means cannot show.
(2) fig_corracc    -- correlation-estimate accuracy by integration pathway, the
                      mechanism the corrected diagnostics support.
(3) fig_decode     -- the decode correction on identical weights: the "~0.45 m
                      floor" against what the same checkpoint gives once decoded
                      query-relative at the true stride.

Encoded by position, marker and grey level, never hue: this venue is commonly
read in print.
"""
import json, os, sys
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

plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 8.5, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
FIGDIR = os.path.join(HERE, 'paper', 'pr', 'figures')

def S(f, st=None):
    t = json.load(open(os.path.join(HERE, 'test_evals', f + '.json')))['tracklets']
    i, o = np.arange(len(t)), np.ones(len(t))
    g, l, n = tracklet_hist(t, st)
    return auc(g, l, n, i, o)[0]

ARMS = [('base', 'Implicit', C_BASE, 'o'), ('forced', 'Additive', C_ADD, 's'),
        ('gated', 'Gated', C_GATE, '^'), ('evid', 'Evidence', C_EVID, 'D')]


def fig_seeds():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, (pre, ns, title) in zip(axes, [('m_', 3, 'KITTI Car (3 seeds)'),
                                           ('nus_', 5, 'nuScenes Car (5 seeds)')]):
        for i, (arm, lab, col, mk) in enumerate(ARMS):
            v = [S(f'{pre}{arm}_s{k}') for k in range(ns)]
            ax.plot([i] * ns, v, mk, ms=5.5, mfc=col, mec=INK, mew=0.8, alpha=.95, zorder=3)
            ax.plot([i - 0.26, i + 0.26], [np.mean(v)] * 2, '-', color=col, lw=2.4, zorder=4)
        ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _, _ in ARMS], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', color=LIGHT, lw=0.4, alpha=.6); ax.set_axisbelow(True)
        ax.margins(y=0.14)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('Success')
    axes[1].legend(handles=[Line2D([], [], marker='o', ls='none', mfc=MID, mec=INK, mew=.8, ms=5.5, label='one run'),
                            Line2D([], [], color=MID, lw=2.4, label='arm mean')],
                   fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/seeds.{e}', bbox_inches='tight', dpi=200)
    print('wrote seeds')


def fig_corracc():
    D = lambda f: json.load(open(os.path.join(HERE, 'diag_fixed', f + '.json')))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    arms = [('forced', 'Additive', C_ADD, 's'), ('gated', 'Gated', C_GATE, '^'),
            ('evid', 'Evidence-only', C_EVID, 'D')]
    for i, (arm, lab, col, mk) in enumerate(arms):
        d = [D(f'corr_m_{arm}_s{s}') for s in range(3)]
        err = [x['inband']['softargmax_err'] for x in d]
        rec = [x['inband']['recall_top1_1cell'] for x in d]
        a1.plot([i] * 3, err, mk, ms=5.5, mfc=col, mec=INK, mew=0.8, zorder=3)
        a1.plot([i - .26, i + .26], [np.mean(err)] * 2, '-', color=col, lw=2.4, zorder=4)
        a2.plot([i] * 3, rec, mk, ms=5.5, mfc=col, mec=INK, mew=0.8, zorder=3)
        a2.plot([i - .26, i + .26], [np.mean(rec)] * 2, '-', color=col, lw=2.4, zorder=4)
    ref = D('corr_car_corr2_SAMEMODEL')
    a1.axhline(0.60, color=LIGHT, ls='--', lw=1)
    a1.text(-0.45, 0.615, 'feature stride', fontsize=6.5, color=MID, ha='left')
    a1.axhline(0.241, color=LIGHT, ls=':', lw=1)
    a1.text(-0.45, 0.256, 'static GT motion', fontsize=6.5, color=MID, ha='left')
    a2.axhline(ref['inband']['fft_f1_64'], color=LIGHT, ls='--', lw=1)
    a2.text(-0.45, ref['inband']['fft_f1_64'] + 2.5, 'FFT occupancy oracle', fontsize=6.5, color=MID, ha='left')
    for ax, ylab in ((a1, 'estimate error, moving (m)'), (a2, 'top-1 within one cell (%)')):
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _, _ in arms], fontsize=8)
        ax.set_xlim(-0.5, 2.5)
        ax.grid(axis='y', color=LIGHT, lw=0.4, alpha=.6); ax.set_axisbelow(True)
        ax.margins(y=0.16)
        ax.set_ylabel(ylab)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/corracc.{e}', bbox_inches='tight', dpi=200)
    print('wrote corracc')


def fig_decode():
    old = json.load(open(os.path.join(HERE, 'diag_corr.json')))
    new = json.load(open(os.path.join(HERE, 'diag_fixed', 'corr_car_corr2_SAMEMODEL.json')))
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    groups = [('static', 'static frames'), ('inband', 'moving frames')]
    x = np.arange(2)
    o = [old[g]['softargmax_err'] for g, _ in groups]
    n = [new[g]['softargmax_err'] for g, _ in groups]
    z = [old[g]['mean_gt_disp'] for g, _ in groups]
    ax.bar(x - 0.21, o, 0.42, color='white', edgecolor=INK, lw=1.0, hatch='///', label='superseded decode')
    ax.bar(x + 0.21, n, 0.42, color=C_BASE, edgecolor=INK, lw=0.8,
           label='corrected decode')
    ax.axhline(0.453, color=MID, ls='--', lw=1)
    ax.text(1.47, 0.465, "decoder's zero-lag bias, 0.453 m", fontsize=6.5, color=MID, ha='right')
    for xi, zi in zip(x, z):
        ax.plot([xi - 0.44, xi + 0.44], [zi, zi], '-', color=MID, lw=1.2, zorder=5)
        ax.text(xi + 0.44, zi + 0.018, f'zero-motion prior {zi:.2f} m',
                fontsize=6.5, color=MID, ha='right')
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel('correlation estimate error (m)')
    ax.set_ylim(0, 1.02)
    ax.grid(axis='y', color=LIGHT, lw=0.4, alpha=.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/decode.{e}', bbox_inches='tight', dpi=200)
    print('wrote decode')


if __name__ == '__main__':
    fig_seeds(); fig_corracc(); fig_decode()
