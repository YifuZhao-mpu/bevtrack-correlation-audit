_base_ = './car_cos.py'

# Corrected protocol (RESEARCH_REVIEW_2026-07-18 r2; checkpoint policy fixed r3).
# - Validation runs on the Valid split (KITTI scenes 17-18); Test (19-20) is held
#   out for ONE frozen evaluation per arm/seed.
# - REPORTING CHECKPOINT (preregistered, coherent single rule): the fixed terminal
#   cosine checkpoint epoch_500.pth. Selection-free — no best-on-Valid selection is
#   used for any reported Test number. (Round 3 flagged the earlier save_best='success'
#   comment as contradicting the epoch-500 Test evals; resolved in favour of the fixed
#   terminal checkpoint, consistent with the paper's terminal-reporting thesis.)
# - The last-30 best-Valid checkpoints are retained ONLY for the C6 checkpoint-
#   sensitivity appendix (a protocol-sensitivity analysis, not a selection rule).
# - primary arms carry NO drift augmentation (sampler default is off).
# Pre-registered decision rules: paired seeds; win/loss = |mean paired dSuccess| >= 1.0
# with 95% tracklet-cluster bootstrap CI excluding 0; parity = ENTIRE 90% CI inside
# +-1 ROPE; third seed whenever |mean| < 1, signs disagree, or Valid/Test rankings differ.
val_dataloader = dict(
    dataset=dict(dataset=dict(split='Valid')),
)

# save_best kept for the retained-checkpoint appendix ONLY; epoch_500 is the reporting ckpt.
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=30, save_best='success', rule='greater'),
)
