"""Apply Codex's revised M1-lite pilot gate from two final.json files (implicit vs se2trans).
Gate (Car): GO if in-band Success >=+4.0 AND in-band Precision >=+2.0 AND overall(all) not worse than -0.3.
Strong GO: + overall(all) Success >=+0.5. NO-GO: in-band Success gain <+2.0 OR overall regress >0.5.
"""
import sys, json

impl = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/home/zyf/paper/refine-logs/m1_implicit_Car/final.json'))['final']
se2 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else '/home/zyf/paper/refine-logs/m1_se2trans_Car/final.json'))['final']


def row(name, a, b):
    return f"{name:9s} implicit {a['success']:6.2f}/{a['precision']:6.2f}  se2trans {b['success']:6.2f}/{b['precision']:6.2f}  Δ {b['success']-a['success']:+6.2f}/{b['precision']-a['precision']:+6.2f}  (n={a['n']})"

print("stratum     implicit S/P        se2trans S/P        Δ S/P            n")
for k in ('all', 'static', 'inband', 'outcrop'):
    print(row(k, impl[k], se2[k]))

dS_in = se2['inband']['success'] - impl['inband']['success']
dP_in = se2['inband']['precision'] - impl['inband']['precision']
dS_all = se2['all']['success'] - impl['all']['success']
print(f"\nin-band ΔSuccess={dS_in:+.2f}  in-band ΔPrecision={dP_in:+.2f}  overall ΔSuccess={dS_all:+.2f}")
print(f"(published BEVTrack Car ref = 64.31/71.14; both heads here = surrogate-encoder, equal budget)")

if dS_in < 2.0 or dS_all < -0.5:
    v = "NO-GO (abandon / reconsider direction)"
elif dS_in >= 4.0 and dP_in >= 2.0 and dS_all >= 0.5:
    v = "STRONG GO"
elif dS_in >= 4.0 and dP_in >= 2.0 and dS_all >= -0.3:
    v = "GO"
else:
    v = "BORDERLINE (in-band win but weak/ flat overall — check calibration, extend budget, or reframe)"
print(f"\nVERDICT: {v}")
