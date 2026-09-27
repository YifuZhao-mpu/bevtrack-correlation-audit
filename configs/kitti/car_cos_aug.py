_base_ = './car_cos.py'

# Control run: baseline + drift_aug only. Attributes any v4b gain between the
# augmentation (helps any head) and the attention-template architecture.
train_dataloader = dict(dataset=dict(cfg=dict(drift_aug=1.5)))
