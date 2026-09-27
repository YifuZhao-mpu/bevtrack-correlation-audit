_base_ = './car_cos.py'

# v4d: R1 exactly as the method section states it -- evidence integration
# (no forced pathway) + direct aux L1 (lambda=0.5) on the matcher, original
# center-cell matcher, NO drift aug. Fair pairs: corr4a (same minus aux),
# cos500 (baseline). Fills the R1 cell corr4a missed (it ran with aux_w=0).
model = dict(
    fuser=dict(type='BEVCorrFuser'),
    head=dict(type='CorrHeadEvidence', aux_w=0.5),
)
