_base_ = './car_cos_valid.py'
# Official-arch third seed (round-3: Valid/Test ranking reversal fired the trigger; submission-blocking).
model = dict(backbone=dict(type='VoxelNetSpconv'))
randomness = dict(seed=2)
