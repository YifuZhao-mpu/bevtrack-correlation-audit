_base_ = './ped_cos_valid.py'
model = dict(fuser=dict(type='BEVCorrFuser', stride_m=0.16), head=dict(type='CorrHeadEvidence'))
randomness = dict(seed=2)
