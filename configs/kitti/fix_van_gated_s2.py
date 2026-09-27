_base_ = './van_cos_valid.py'
model = dict(fuser=dict(type='BEVCorrFuser', stride_m=0.6), head=dict(type='CorrHeadGated'))
randomness = dict(seed=2)
