_base_ = './car_cos.py'

# Seed-2 replica of cos500 (set SEED=2 via env at launch). Separate file only
# so the work_dir (derived from config basename) doesn't clobber the seed-1 run.
