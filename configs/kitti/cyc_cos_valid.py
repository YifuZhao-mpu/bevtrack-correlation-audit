_base_ = './cyc_cos.py'
# Category sweep for the journal (PR) route. Cyclist Valid is TINY (2 tracklets/101 frames)
# — monitoring only; reporting checkpoint is the fixed terminal epoch_500.
# Cyclist voxel 0.03 -> feature stride 0.03*2^3 = 0.24 m/cell: corr fusers MUST pass
# stride_m=0.24 (Car is 0.6, Pedestrian 0.16) or the decode-bias bug reappears.
val_dataloader = dict(dataset=dict(dataset=dict(split='Valid')))
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=30, save_best='success', rule='greater'))
