"""M0 step 3: real nuScenes data through the pipeline (keyframe-only data).
Instantiate the SOT loader on local data, pull a Car tracklet, run TrainSampler.processing
on a consecutive pair, print shapes + the (dx,dy,dz,dyaw) label."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mmengine.config import ConfigDict
from datasets.nuscenes import NuScenesDataset
from datasets.sampler import TrainSampler

DATA = '/home/zyf/data/nuscenes'

print("instantiating NuScenes val Car loader (no preloading)...")
ds = NuScenesDataset(path=DATA, split='val', category_name='Car', preloading=False)
print(f"  #tracklets={ds.get_num_tracklets()}  #frames_total={ds.get_num_frames_total()}")
assert ds.get_num_tracklets() > 0

# pull first tracklet, first two frames
frames = ds.get_frames(0, [0, 1])
print(f"  frame0 pts={frames[0]['pc'].points.shape}  box.center={np.round(frames[0]['3d_bbox'].center,2)} wlh={np.round(frames[0]['3d_bbox'].wlh,2)}")
print(f"  frame1 pts={frames[1]['pc'].points.shape}")

cfg = ConfigDict(dict(num_candidates=2, target_thr=None, search_thr=5,
                      point_cloud_range=[-4.8, -4.8, -1.5, 4.8, 4.8, 1.5],
                      time_flip=True, flip=True))
data = {'prev_frame': frames[0], 'this_frame': frames[1], 'candidate_id': 0}
out = TrainSampler.processing(data, cfg)
inp, ds_s = out['inputs'], out['data_samples']
print(f"  prev_points={tuple(inp['prev_points'].shape)}  this_points={tuple(inp['this_points'].shape)}")
print(f"  box_label (dx,dy,dz,dyaw_deg) = {np.round(ds_s['box_label'].numpy(),4)}")
print(f"  wlh = {np.round(inp['wlh'].numpy(),3)}")
assert inp['prev_points'].shape[1] == 3 and ds_s['box_label'].shape[0] == 4
print("M0_DATA_PASS")
