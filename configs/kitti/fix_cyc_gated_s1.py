_base_ = './cyc_cos_valid.py'
model = dict(fuser=dict(type='BEVCorrFuser', stride_m=0.24), head=dict(type='CorrHeadGated'))
randomness = dict(seed=1)
