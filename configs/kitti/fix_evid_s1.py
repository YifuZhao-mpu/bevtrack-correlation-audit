_base_ = './car_cos_valid.py'
# Corrected matrix, evidence arm (corr disp+conf as features only), seed 1.
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadEvidence'))
randomness = dict(seed=1)
