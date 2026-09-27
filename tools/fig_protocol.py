"""Figure: the checkpoint lottery, and what fixing the protocol does and does not fix.

(a) The codebase's official recipe -- 800 epochs at constant LR -- read from the
    per-epoch validation curves it produced. Raw against a 9-epoch trend, with
    the best checkpoint marked: that is the number a best-checkpoint convention
    would report, and it sits well above the run's own trend.
(b) Three seeds of the matched baseline under the fixed protocol. These runs
    validate on the Valid split rather than Test, so their within-run spans are
    NOT comparable with (a); the panel is here for the other lottery, that three
    runs differing only in seed end apart.

Both curves in (a) validate on the same split, which is what makes the
comparison of schedules meaningful. An earlier draft of this script put the
matched Valid-split runs beside the legacy Test-split ones and read the
difference as annealing removing the scatter. It is not: on a common split the
last-ten span goes 3.88 -> 3.28 and the adjacent-epoch median 1.47 -> 1.51.
"""
import csv, json, glob
from pathlib import Path
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf, np.NINF = np.inf, -np.inf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/zyf/paper/BEVTrack')
INK, MID, LIGHT = '#1a1a1a', '#6e6e6e', '#b4b4b4'
# validated categorical palette, same fixed slot order as the other figures
C_BASE, C_ADD, C_GATE, C_EVID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 8.5, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})


def legacy(name):
    with open(ROOT / name) as f:
        r = [(int(x['epoch']), float(x['success'])) for x in csv.DictReader(f) if x['success']]
    r.sort()
    return np.array([e for e, _ in r]), np.array([s for _, s in r])


def matched(seed):
    pts = {}
    for f in sorted(glob.glob(str(ROOT / f'work_dir/m_base_s{seed}/*/vis_data/scalars.json'))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if 'success' in d and 'step' in d:
                pts[int(d['step'])] = float(d['success'])
    e = np.array(sorted(pts))
    return e, np.array([pts[k] for k in e])


def trend(y, w=9):
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

# ---- (a) the official constant-LR recipe ----
e, y = legacy('none800_car_curve.csv')
ax1.plot(e, y, '-', color=C_BASE, lw=0.5, alpha=.35)
t = trend(y)
m = (e > 4) & (e < e.max() - 4)
ax1.plot(e[m], t[m], '-', color=C_BASE, lw=1.8, label='constant LR, 800 ep')
e3, y3 = legacy('cos500_car_curve.csv')
t3 = trend(y3)
m3 = (e3 > 4) & (e3 < e3.max() - 4)
ax1.plot(e3, y3, '-', color=C_ADD, lw=0.5, alpha=.35)
ax1.plot(e3[m3], t3[m3], '-', color=C_ADD, lw=1.8, label='cosine to $10^{-6}$, 500 ep')
i = int(np.argmax(y))
ax1.plot([e[i]], [y[i]], 'o', ms=6, mfc=C_ADD, mec=INK, mew=0.8, zorder=5)
ax1.annotate(f'best checkpoint {y[i]:.1f}, {y[i]-t[i]:+.1f}\nover its own 9-epoch trend',
             xy=(e[i], y[i]), xytext=(e[i] - 300, y[i] - 7.6), fontsize=7, color=INK,
             ha='center', arrowprops=dict(arrowstyle='-', color=MID, lw=0.7, shrinkB=5))
ax1.text(0.02, 0.03, 'last-ten span 3.88 vs 3.28;\nadjacent-epoch median 1.47 vs 1.51',
         transform=ax1.transAxes, fontsize=6.5, color=MID, va='bottom')
ax1.set_ylim(63.5, 75.5)
ax1.set_xlabel('epoch'); ax1.set_ylabel('validation Success')
ax1.set_title('(a) both schedules, same validation split', fontsize=9)
ax1.legend(fontsize=7, frameon=False, loc='upper right')

# ---- (b) the fixed protocol, three seeds ----
cols = [C_BASE, C_ADD, C_GATE]
finals = []
for s in range(3):
    e2, y2 = matched(s)
    ax2.plot(e2, y2, '-', color=cols[s], lw=0.7, alpha=.85, label=f'seed {s}')
    finals.append(y2[-1])
ax2.set_xlabel('epoch'); ax2.set_title('(b) three seeds, Valid split', fontsize=9)
ax2.legend(fontsize=7, frameon=False, loc='lower right', ncol=3, columnspacing=1.0)
lo, hi = min(finals), max(finals)
ax2.set_ylim(63.5, 75.5)
ax2.annotate(f'three runs end {hi - lo:.2f} apart',
             xy=(500, (lo + hi) / 2), xytext=(300, (lo + hi) / 2 - 5.4), fontsize=7,
             color=INK, ha='center',
             arrowprops=dict(arrowstyle='-', color=MID, lw=0.7, shrinkB=4))

for ax in (ax1, ax2):
    ax.grid(color=LIGHT, lw=0.4, alpha=.6); ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(ROOT / f'paper/pr/figures/protocol_curves.{ext}', bbox_inches='tight', dpi=200)
    print('wrote', ext)
print('legacy best %.4f trend %.4f spike %+.4f' % (y[i], t[i], y[i] - t[i]))
print('matched finals', ['%.2f' % f for f in finals], 'spread %.2f' % (hi - lo))
