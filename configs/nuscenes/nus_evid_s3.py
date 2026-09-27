_base_ = './car_cos_valid.py'
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadMatched', inject='none'))
randomness = dict(seed=3)
