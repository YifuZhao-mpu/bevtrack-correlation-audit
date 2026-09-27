_base_ = './car_cos.py'

# v4c: coarse-to-fine explicit matching. Coarse 16x16 attn-template correlation
# (drift-robust anchor) + fine 64x64 (0.15 m/cell) windowed refinement -- attacks
# the feature-stride floor diag_refine.json exposed (no 16x16 decode beats ~0.46m).
# Same evidence head + aux supervision + drift_aug as v4b; only the fuser deepens.
model = dict(
    backbone=dict(return_pyramid=True),
    fuser=dict(type='BEVCorrFuserC2F'),
    head=dict(type='CorrHeadEvidence', aux_w=0.5, in_extra=6),
)

train_dataloader = dict(dataset=dict(cfg=dict(drift_aug=1.5)))
