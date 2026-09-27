_base_ = './ped_cos.py'
# Second-category breadth (round-3 gate 6). Same corrected protocol as car_cos_valid:
# Valid-split validation (scenes 17-18), epoch_500 reporting checkpoint, retain-30 for
# the C6 appendix, no drift-aug. Pedestrian feature stride = 0.02 voxel x 2^3 = 0.16 m/cell
# (NOT Car's 0.6) — corr fusers below MUST pass stride_m=0.16 or the decode bug reappears.
val_dataloader = dict(dataset=dict(dataset=dict(split='Valid')))
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=30, save_best='success', rule='greater'))
