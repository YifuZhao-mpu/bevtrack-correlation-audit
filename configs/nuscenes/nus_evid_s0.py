_base_ = './car_cos_valid.py'
# matched integration head: identical residual inputs/capacity, only the injection rule differs
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadMatched', inject='none'))
randomness = dict(seed=0)
