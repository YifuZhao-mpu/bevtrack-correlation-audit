_base_ = './car_cos_valid.py'
# Third seed (pre-registered trigger 2026-07-18: base-vs-forced signs disagree at 2 seeds).
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHead'))
randomness = dict(seed=2)
