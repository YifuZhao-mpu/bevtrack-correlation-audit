_base_ = './car_cos_valid.py'
# Third seed (pre-registered trigger: evidence-vs-base signs disagree at 2 seeds).
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadEvidence'))
randomness = dict(seed=2)
