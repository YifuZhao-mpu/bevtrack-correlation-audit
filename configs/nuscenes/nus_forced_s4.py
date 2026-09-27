_base_ = './car_cos_valid.py'
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadMatched', inject='additive'))
randomness = dict(seed=4)
