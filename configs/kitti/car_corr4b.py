_base_ = './car_cos.py'

# v4b: drift-robust explicit matching. Attention-pooled template (no center-cell
# cliff) + evidence-only head (no forced pathway) + drift_aug 1.5 m so the drifted
# regime is trained. Controls: cos500 (baseline), corr4a (evidence w/o attn+aug),
# car_cos_aug (baseline + same aug, isolates the aug factor).
model = dict(
    fuser=dict(type='BEVCorrFuserAttn'),
    head=dict(type='CorrHeadEvidence', aux_w=0.5),
)

train_dataloader = dict(dataset=dict(cfg=dict(drift_aug=1.5)))
