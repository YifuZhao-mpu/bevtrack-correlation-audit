_base_ = './car_cos_valid.py'
# Corrected matrix, forced-additive full-capacity arm (corrected decode), seed 0.
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHead'))
randomness = dict(seed=0)
