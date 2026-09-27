# Explicit Correlation Is Not a Drop-In Prior

Code, frozen per-tracklet results and the statistical tooling for a controlled
audit of explicit bird's-eye-view (BEV) correlation in 3D single object
tracking.

> **Built on the official BEVTrack implementation**
> ([xmm-prio/BEVTrack](https://github.com/xmm-prio/BEVTrack), MIT,
> © 2024 xmm-prio). The upstream README is kept as
> [`README_BEVTrack_upstream.md`](README_BEVTrack_upstream.md). This repository
> adds the three integration pathways, the matched-capacity head, the
> per-tracklet evaluator, the counterfactual harness and the two-level interval
> estimator described below. The original licence and copyright are retained in
> [`LICENSE`](LICENSE).

## What the audit asks

Inter-frame target motion in 3D SOT is close to a pure 2D translation, so
explicit BEV correlation is the canonical estimator and the obvious thing to
add to a global-regression tracker that discards spatial structure. This
repository tests whether it earns its place.

Three integration pathways are implemented as **one head with one flag**
(`inject` in `models/corr_head.py`):

| `inject` | pathway | the correlation estimate … |
|---|---|---|
| `'none'` | evidence-only | enters the head as an input, never reaches the output |
| `'additive'` | additive injection | is added to the output |
| `'gated'` | gated injection | is added, scaled by a gate predicted from `f` and `c` |

All three share the same residual MLP and the same inputs
(`f ⊕ d̂ ⊕ c`), matched in capacity to within 0.2% of the model, so a
difference between them is attributable to the pathway rather than to the
information or the parameters available to the head.

## What it found

Across 32 training runs on KITTI and nuScenes:

- **No pathway produces a resolved gain** over implicit regression, on either
  dataset or displacement stratum, nor — for the two arms retrained on it — on
  a second encoder.
- **Gated injection is resolved harm on the KITTI surrogate encoder**
  (−3.40 Success overall, −4.79 on moving targets), but reverses to +1.41 and
  becomes unresolved once the official sparse-convolution encoder is restored.
- A closed-loop analysis locates that harm in **a template that collapses once
  the tracker's own drift exceeds one BEV cell**, not in the quality of the
  correlation estimate, which matches an FFT oracle under oracle references.
- **Retraining an identical configuration moves Success by σ = 1.20 (KITTI) to
  2.00 (nuScenes)** — a component that confidence intervals over test items
  exclude by construction. On nuScenes such an interval understated the
  seed-to-seed spread by more than twelvefold.
- **Reliance is not contribution.** Across all eight evidence-only models,
  shuffling the correlation branch in-distribution costs up to 15.8 Success,
  yet that reliance is uncorrelated with how much the model gains from the
  branch (r = −0.02, n = 8).

### The decision rule

A contrast is reported as **resolved** only when all three hold:

1. |Δ| ≥ 1.0 Success, **and**
2. the two-level (seed × cluster) bootstrap interval excludes zero, **and**
3. the seed-level *t* interval agrees.

`tools/seed_scene_ci.py` implements exactly this rule. Until 2026-09-23 it
consulted only condition 2, which labels four comparisons resolved that the
rule does not; the paper's verdicts were applied by hand and are unaffected.
The fix is in the file's own comments.

## The three released tools

| tool | what it does |
|---|---|
| `tools/eval_tracklets.py` | per-tracklet evaluator; emits the frozen JSONs in `test_evals/` |
| `tools/mech_evidence.py` | evidence-pathway counterfactual: replaces `d̂` at inference with a draw from the same model's own rollout, leaving weights untouched |
| `tools/seed_scene_ci.py` | two-level (seed × cluster) interval estimator, with `tools/scene_bootstrap.py` and `tools/paired_bootstrap.py` |

## Layout

```
models/ datasets/ configs/   architecture, data pipeline, run configs
tools/                       evaluator, diagnostics, counterfactuals, figures
test_evals/                  85 frozen per-tracklet eval JSONs (the paper's numbers)
diag_fixed/                  corrected correlation diagnostics (2026-09-23)
mech_ev/ mech_cl/ mech_dec/  counterfactual, closed-loop and decomposition runs
run_*.sh                     the training / evaluation / interval drivers
intervals_final.txt          every printed interval, recomputed at fixed draws
FINAL_NUMBERS.txt            every reported number, from the frozen JSONs
```

## Reproducing the intervals

The interval recomputation needs only the frozen JSONs — no GPU, no
checkpoints. The estimator is seeded and the draw counts are fixed, so it
reproduces the endpoints exactly:

```bash
PY=python ./run_ci_final.sh      # writes intervals_final.txt
```

A single contrast, for example KITTI gated against the implicit baseline:

```bash
python tools/seed_scene_ci.py \
  test_evals/m_base_s0.json,test_evals/m_base_s1.json,test_evals/m_base_s2.json \
  test_evals/m_gated_s0.json,test_evals/m_gated_s1.json,test_evals/m_gated_s2.json \
  --unit tracklet --n-boot 40000
```

Training and evaluation follow the upstream BEVTrack instructions; see
`README_BEVTrack_upstream.md` for dataset preparation, and `configs/` for the
matched-head variants.

### Two things that will not run as-is

- **Dataset paths are hardcoded.** The `configs/` files and most `run_*.sh`
  drivers carry absolute paths from the machine the experiments ran on. Point
  them at your own KITTI / nuScenes roots before training.
- **The figure scripts write into the paper tree.** `tools/fig_*.py` and
  `tools/make_assets.py` save to `paper/pr/figures/`, which is not part of this
  repository. They are included so the figures are auditable, not because they
  run out of the box; create that directory or edit the output path.

Some `run_*.sh` drivers are superseded — they predate the coordinate-decode
correction and the matched head, and are kept as the experimental record rather
than as a reproduction path. The current path is `run_pr_matched.sh`,
`run_pr_nus.sh`, `run_diag_fixed.sh`, `run_diag2_fixed.sh` and
`run_ci_final.sh`.

## Not included

- **Checkpoints and `work_dir/`** (≈384 GB) are regenerable from the configs
  and the seeds recorded in the run scripts.
- **Raw training logs.** The distilled per-tracklet results in `test_evals/`
  are what the paper's numbers are computed from, and those are included.

## A superseded artifact, kept on purpose

`diag_corr.json` is the **pre-correction** correlation diagnostic. Its decode
placed zero lag at (+0.32, +0.32) m on a 0.64 m grid, giving the decoder a
0.453 m zero-lag bias; an earlier draft read that bias as a "feature-stride
floor" and the paper withdraws the claim. The file is retained because
`tools/fig_results.py::fig_decode()` plots it against the corrected
diagnostic in `diag_fixed/` — the paper shows the correction rather than
quietly replacing it.

## Citation

The paper is under review. Please cite the upstream BEVTrack work for the
baseline architecture:

```bibtex
@inproceedings{yang2023bevtrack,
  author    = {Yang, Yuxiang and Deng, Yingqi and Pan, Mian and Zha, Zheng-Jun and Zhang, Jing},
  title     = {{BEVTrack}: A Simple and Strong Baseline for {3D} Single Object
               Tracking in Bird's-Eye View},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference
               on Artificial Intelligence (IJCAI)},
  pages     = {2242--2250},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/250}
}
```

## Licence

MIT, inherited from the upstream project — see [`LICENSE`](LICENSE).
