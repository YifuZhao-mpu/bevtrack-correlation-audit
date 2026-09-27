_base_ = './car_cos_valid.py'
# Corrected matrix, gated arm (corrected decode), seed 0.
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadGated'))
randomness = dict(seed=0)
