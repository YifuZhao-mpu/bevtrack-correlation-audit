_base_ = './car.py'

# Protocol fix: every 800ep constant-lr run (none/disp/corr3) degrades after its
# mid-run peak, so final-epoch numbers are unusable and best-checkpoint selection
# is noisy. Cosine decay to ~0 makes final == best and halves the budget.
epoch_num = 500

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=epoch_num, eta_min=1e-6,
         begin=0, end=epoch_num, by_epoch=True, convert_to_iter_based=True)
]
