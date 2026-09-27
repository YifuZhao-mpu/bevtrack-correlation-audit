_base_ = './car_cos_valid.py'
# Corrected matrix, evidence arm (corr disp+conf as features only), seed 0.
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadEvidence'))
randomness = dict(seed=0)
