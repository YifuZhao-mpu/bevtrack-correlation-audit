_base_ = './car_corr4a.py'

# v4e: corr4a + drift aug recalibrated to coverage, not domination
# (p=0.1, offsets 0.5..1.0 m vs the harmful p=0.5, 0.3..1.5 m of v4b/v4c;
# cosaug showed the old dose costs the plain baseline 4.3 terminal points).
# Fair pair: corr4a (identical model, aug only).
train_dataloader = dict(dataset=dict(cfg=dict(
    drift_aug=1.0, drift_aug_p=0.1, drift_aug_min=0.5)))
