_base_ = './ped.py'

# Fixed protocol (see car_cos.py): cosine to ~0 makes final == best.
epoch_num = 500

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num, eta_min=1e-6,
         begin=0, end=epoch_num, by_epoch=True, convert_to_iter_based=True)
]
