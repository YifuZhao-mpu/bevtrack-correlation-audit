"""M0 step 1: geometry convention round-trip (no data, no GPU).
Validates transform_box (label) <-> getOffsetBB (inference) are exact inverses,
and that box_label = (dx,dy,dz, signed-yaw-deg) of this_box in ref/base local frame.
This is the exact convention the SE(2) head must reproduce.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from datasets.points_utils import getOffsetBB, transform_box


def make_box(center, yaw_deg, wlh=(1.9, 4.5, 1.7)):
    return Box(list(center), list(wlh), Quaternion(axis=[0, 0, 1], degrees=yaw_deg))


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


np.random.seed(0)
max_c_err = 0.0
max_a_err = 0.0
N = 2000
for _ in range(N):
    ref = make_box(np.random.uniform(-30, 30, 3), np.random.uniform(-180, 180))
    dx, dy, dz = np.random.uniform(-3, 3, 3)
    dth = np.random.uniform(-28, 28)
    # forward: build this_box by applying a known offset in ref's local frame
    this = getOffsetBB(ref, [dx, dy, dz, dth], degrees=True, use_z=True, limit_box=False)
    # label: express this_box in ref frame
    lab = transform_box(this, ref)
    rec_c = np.array(lab.center)
    rec_deg = lab.orientation.degrees * lab.orientation.axis[-1]
    # the label must equal the offset we applied
    max_c_err = max(max_c_err, np.abs(rec_c - np.array([dx, dy, dz])).max())
    max_a_err = max(max_a_err, abs(wrap180(rec_deg - dth)))
    # round-trip: reconstruct this_box from (ref, label) as inference does
    rt = getOffsetBB(ref, [rec_c[0], rec_c[1], rec_c[2], rec_deg],
                     degrees=True, use_z=True, limit_box=False)
    assert np.allclose(rt.center, this.center, atol=1e-4), (rt.center, this.center)

print(f"trials={N}  max center err={max_c_err:.2e} m  max yaw err={max_a_err:.2e} deg")
assert max_c_err < 1e-3 and max_a_err < 1e-2, "convention mismatch!"
print("M0_GEOMETRY_PASS")
