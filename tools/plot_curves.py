"""Plot per-stratum Success vs training step from eval_log.json (convergence check + paper fig)."""
import sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

runs = sys.argv[1:] or ['/home/zyf/paper/refine-logs/m1_implicit_Car', '/home/zyf/paper/refine-logs/m1_se2trans_Car']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
strata = ['all', 'static', 'inband']
for run in runs:
    try:
        log = json.load(open(f'{run}/eval_log.json'))
    except FileNotFoundError:
        continue
    name = run.split('/')[-1].replace('m1_', '')
    steps = [e['step'] for e in log]
    for ax, s in zip(axes, strata):
        ax.plot(steps, [e[s]['success'] for e in log], marker='o', label=name)
        ax.set_title(f'{s} Success'); ax.set_xlabel('step'); ax.grid(alpha=.3)
axes[0].legend()
plt.tight_layout(); plt.savefig('/home/zyf/paper/refine-logs/m1_curves.png', dpi=110)
print("wrote /home/zyf/paper/refine-logs/m1_curves.png")
