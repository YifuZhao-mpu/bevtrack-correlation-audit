_base_ = './van_cos.py'
# Category sweep for the journal (PR) route. Same corrected protocol as car_cos_valid:
# Valid-split monitoring only (Van Valid is TINY: 3 tracklets/59 frames — never used for
# selection; the preregistered reporting checkpoint is the fixed terminal epoch_500).
# Van voxel 0.075 -> feature stride 0.6 m (same as Car; corr fusers use the default).
val_dataloader = dict(dataset=dict(dataset=dict(split='Valid')))
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=30, save_best='success', rule='greater'))
