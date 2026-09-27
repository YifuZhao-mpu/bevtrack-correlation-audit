import torch
from torch import nn
from mmengine.registry import MODELS

GN = 8


@MODELS.register_module()
class BEVFuser(nn.Sequential):
    """Implicit localizer: cat(prev,this) -> conv stack -> global maxpool -> 512 vec.
    GroupNorm (was SyncBatchNorm) for single-GPU stability / no running-stat eval drift."""

    def __init__(self):
        super().__init__(
            ConvReluGN(256, 256), ConvReluGN(256, 256),
            ConvReluGN(256, 512, 2), ConvReluGN(512, 512), ConvReluGN(512, 512),
            ConvReluGN(512, 512, 2), ConvReluGN(512, 512), ConvReluGN(512, 512),
            nn.AdaptiveMaxPool2d(1), nn.Flatten())

    def forward(self, stack_feats):
        cat_feats = torch.cat(stack_feats.chunk(2, 0), 1)
        return super().forward(cat_feats)


class ConvReluGN(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(GN, cout),
            nn.ReLU(True))
