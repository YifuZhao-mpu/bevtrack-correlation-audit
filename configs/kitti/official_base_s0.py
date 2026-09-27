_base_ = './car_cos_valid.py'
# Official sparse-3D architecture confirmation run (review MUST #4): baseline arm, seed 0.
model = dict(backbone=dict(type='VoxelNetSpconv'))
randomness = dict(seed=0)
