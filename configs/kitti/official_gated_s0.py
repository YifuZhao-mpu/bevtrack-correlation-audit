_base_ = './car_cos_valid.py'
# Official sparse-3D architecture confirmation run (review MUST #4): gated arm, seed 0.
model = dict(backbone=dict(type='VoxelNetSpconv'),
             fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadGated'))
randomness = dict(seed=0)
