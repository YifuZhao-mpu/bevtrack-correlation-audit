import numpy as np
import torch
import torchmetrics.utilities.data
from shapely.geometry import Polygon
from torchmetrics import Metric
from mmengine.evaluator import BaseMetric
import torchmetrics.utilities.data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def estimateAccuracy(box_a, box_b, dim=3, up_axis=(0, -1, 0)):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        up_axis = np.array(up_axis)
        return np.linalg.norm(
            box_a.center[up_axis != 0] - box_b.center[up_axis != 0], ord=2)


def fromBoxToPoly(box, up_axis=(0, -1, 0)):
    """

    :param box:
    :param up_axis: the up axis must contain only one non-zero component
    :return:
    """
    if up_axis[1] != 0:
        return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))
    elif up_axis[2] != 0:
        return Polygon(tuple(box.bottom_corners().T))


def estimateOverlap(box_a, box_b, dim=2, up_axis=(0, -1, 0)):
    # if box_a == box_b:
    #     return 1.0
    try:
        Poly_anno = fromBoxToPoly(box_a, up_axis)
        Poly_subm = fromBoxToPoly(box_b, up_axis)

        box_inter = Poly_anno.intersection(Poly_subm)
        box_union = Poly_anno.union(Poly_subm)
        if dim == 2:
            return box_inter.area / box_union.area

        else:
            up_axis = np.array(up_axis)
            up_max = min(box_a.center[up_axis != 0], box_b.center[up_axis != 0])
            up_min = max(box_a.center[up_axis != 0] - box_a.wlh[2], box_b.center[up_axis != 0] - box_b.wlh[2])
            inter_vol = box_inter.area * max(0, up_max[0] - up_min[0])
            anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
            subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

            overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
            return overlap
    except ValueError:
        return 0.0


class TorchPrecision(Metric):
    """Computes and stores the Precision using torchMetrics"""

    def __init__(self, n=21, max_accuracy=2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_accuracy = max_accuracy
        self.Xaxis = torch.linspace(0, self.max_accuracy, steps=n)
        self.add_state("accuracies", default=[])

    def value(self, accs):
        prec = [
            torch.sum((accs <= thres).float()) / len(accs)
            for thres in self.Xaxis
        ]
        return torch.tensor(prec)

    def update(self, val):
        self.accuracies.append(val)

    def compute(self):
        accs = torchmetrics.utilities.data.dim_zero_cat(self.accuracies)
        if accs.numel() == 0:
            return 0
        return torch.trapz(self.value(accs), x=self.Xaxis) * 100 / self.max_accuracy


class TorchSuccess(Metric):
    """Computes and stores the Success using torchMetrics"""

    def __init__(self, n=21, max_overlap=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_overlap = max_overlap
        self.Xaxis = torch.linspace(0, self.max_overlap, steps=n)
        self.add_state("overlaps", default=[])

    def value(self, overlaps):
        succ = [
            torch.sum((overlaps >= thres).float()) / len(overlaps)
            for thres in self.Xaxis
        ]
        return torch.tensor(succ)

    def compute(self):
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return 0
        return torch.trapz(self.value(overlaps), x=self.Xaxis) * 100 / self.max_overlap

    def update(self, val):
        self.overlaps.append(val)


class TrackAccuracy(BaseMetric):

    def __init__(self,
                 n=21,
                 max_accuracy=2,
                 max_overlap=1,
                 **kargs):
        super().__init__(**kargs)
        self.max_accuracy = max_accuracy
        self.max_overlap = max_overlap
        self.prec_axis = torch.linspace(0, self.max_accuracy, steps=n)
        self.succ_axis = torch.linspace(0, self.max_overlap, steps=n)

    def process(self, data_batch, data_samples):
        overlap, distance, disp = torch.tensor(data_samples)
        self.results.append((distance, overlap, disp))

    def _stratum(self, distance, overlap):
        if overlap.numel() == 0:
            return dict(success=0.0, precision=0.0, n=0)
        prec = torch.tensor([torch.sum((distance <= t).float()) / len(distance) for t in self.prec_axis])
        succ = torch.tensor([torch.sum((overlap >= t).float()) / len(overlap) for t in self.succ_axis])
        return dict(success=float(torch.trapz(succ, x=self.succ_axis) * 100 / self.max_overlap),
                    precision=float(torch.trapz(prec, x=self.prec_axis) * 100 / self.max_accuracy),
                    n=int(overlap.numel()))

    def compute_metrics(self, results):
        distance, overlap, disp = zip(*results)
        cat = torchmetrics.utilities.data.dim_zero_cat
        distance, overlap, disp = cat(distance), cat(overlap), cat(disp)
        import os as _os
        _sthr = float(_os.environ.get('STRAT_STATIC', 0.5))   # static/moving boundary (m)
        _crop = float(_os.environ.get('STRAT_CROP', 4.8))     # crop half-range (m), category-dependent
        st = (disp < _sthr); ib = (disp >= _sthr) & (disp <= _crop); oc = (disp > _crop)
        strata = dict(all=self._stratum(distance, overlap),
                      static=self._stratum(distance[st], overlap[st]),
                      inband=self._stratum(distance[ib], overlap[ib]),
                      outcrop=self._stratum(distance[oc], overlap[oc]))
        import json, os
        print('\n=== STRATIFIED (BEV displacement) ===')
        for k, v in strata.items():
            print(f"  {k:8s} Success/Precision {v['success']:.2f}/{v['precision']:.2f}  (n={v['n']})")
        tag = os.environ.get('STRAT_TAG', 'strat')
        json.dump({'final': strata}, open(f'/home/zyf/paper/BEVTrack/{tag}.json', 'w'), indent=2)
        return dict(success=strata['all']['success'], precision=strata['all']['precision'])
