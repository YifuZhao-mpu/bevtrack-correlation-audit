_base_ = './car_cos_valid.py'
# Third seed (pre-registered trigger, pending gated 2-seed readout).
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadGated'))
randomness = dict(seed=2)
