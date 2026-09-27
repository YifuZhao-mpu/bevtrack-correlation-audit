_base_ = './car_cos.py'

# v4a: corr evidence as features only (no additive/gated pathway) + cosine-500
# protocol. Fair pair = cos500 baseline (same protocol, same budget).
model = dict(
    fuser=dict(type='BEVCorrFuser'),
    head=dict(type='CorrHeadEvidence'),
)
