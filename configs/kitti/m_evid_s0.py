_base_ = './car_cos_valid.py'
# matched integration head: identical residual inputs (feat+disp+conf) and capacity;
# ONLY the injection rule differs (round-4 req#3 — the legacy CorrHead was unmatched).
model = dict(fuser=dict(type='BEVCorrFuser'), head=dict(type='CorrHeadMatched', inject='none'))
randomness = dict(seed=0)
