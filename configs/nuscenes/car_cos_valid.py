_base_ = './car.py'

# nuScenes protocol for the PR package (review 2026-07-25 round 4, requirement #1:
# an untouched MULTI-SCENE dataset). nuScenes val has 150 scenes, so scene — not
# tracklet — can finally be the resampling unit, which KITTI (2 Test scenes, whole
# categories confined to one) structurally could not support.
# Nothing here is tuned on val: the reporting checkpoint is the fixed terminal one,
# exactly as on KITTI. Validation during training exists only for curve monitoring.
# Feature stride = 0.075 * 2^3 = 0.6 m/cell (same as KITTI Car); corr configs inherit it.
epoch_num = 40   # nuScenes train_track is ~40x KITTI Car; budget set by pilot timing

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=10)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num, eta_min=1e-6,
         begin=0, end=epoch_num, by_epoch=True, convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, save_best='success', rule='greater'))

# --- host-RAM fix (2026-09-12): the nus matrix died three times with exit=137 ---
# Root cause: preload_nuscenes_Car_train_track_*.dat is 9.1 GB and lives in the
# parent process. With mp_start_method='fork' + persistent_workers=True, CPython
# refcount writes break copy-on-write, so each of the 4 workers gradually
# materialises its own copy of the object graph; 5 x 9.1 GB exceeds the 48 GB of
# available RAM and the OOM killer fires around epoch 15 (matches the observed
# 1.5 h time-to-kill on 2026-07-27 01:44).
# Fix: persistent_workers=False re-forks the workers every epoch, which resets the
# COW drift before it can accumulate; num_workers=2 halves the peak. Data content,
# sampling, seeds and the reporting checkpoint are all untouched -- this is a
# loader-parallelism change only and has no effect on the protocol.
train_dataloader = dict(num_workers=2, persistent_workers=False)
val_dataloader = dict(num_workers=2, persistent_workers=False)
