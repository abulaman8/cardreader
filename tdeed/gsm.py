"""Gate-Shift Module (GSM) inserted into timm RegNet blocks, after T-DEED / E2E-Spot.

GSM: Sudhakaran et al., "Gate-Shift Networks for Video Action Recognition" (BSD-2, FBK).
Adapted by Hong et al. (E2E-Spot) and Xarles et al. (T-DEED). Re-implemented here without
torch.cuda.FloatTensor so it runs on any device and under autocast.
"""
import math

import torch
import torch.nn as nn


class GSM(nn.Module):
    """Gated temporal shift on `planes` channels. Input (B*T, C, H, W); T set via `num_segments`."""

    def __init__(self, planes, num_segments=1):
        super().__init__()
        assert planes % 4 == 0, planes
        self.planes, self.num_segments = planes, num_segments
        self.conv3d = nn.Conv3d(planes, 2, (3, 3, 3), stride=1, padding=(1, 1, 1), groups=2)
        nn.init.zeros_(self.conv3d.weight)
        nn.init.zeros_(self.conv3d.bias)
        self.bn = nn.BatchNorm3d(planes)

    def forward(self, x):
        T = self.num_segments
        nt, c, h, w = x.shape
        b = nt // T
        x = x.view(b, T, c, h, w).permute(0, 2, 1, 3, 4).contiguous()          # (B, C, T, H, W)
        gate = torch.tanh(self.conv3d(torch.relu(self.bn(x))))
        g1, g2 = gate[:, 0:1], gate[:, 1:2]
        x1, x2 = x[:, :c // 2], x[:, c // 2:]
        y1, y2 = g1 * x1, g2 * x2
        r1, r2 = x1 - y1, x2 - y2
        y1 = torch.cat([y1[:, :, 1:], torch.zeros_like(y1[:, :, :1])], dim=2) + r1   # from t+1
        y2 = torch.cat([torch.zeros_like(y2[:, :, :1]), y2[:, :, :-1]], dim=2) + r2  # from t-1
        y1 = y1.view(b, 2, c // 4, T, h, w).permute(0, 2, 1, 3, 4, 5).reshape(b, c // 2, T, h, w)
        y2 = y2.view(b, 2, c // 4, T, h, w).permute(0, 2, 1, 3, 4, 5).reshape(b, c // 2, T, h, w)
        y = torch.cat([y1, y2], dim=1)
        return y.permute(0, 2, 1, 3, 4).reshape(nt, c, h, w)


class GatedShift(nn.Module):
    """Applies GSM to the first `fold` input channels, then the wrapped conv."""

    def __init__(self, net, n_div=4):
        super().__init__()
        if hasattr(net, "conv"):
            channels = net.conv.in_channels
        elif isinstance(net, nn.Conv2d):
            channels = net.in_channels
        else:
            raise NotImplementedError(type(net))
        self.fold = math.ceil(channels // n_div / 4) * 4
        self.gs = GSM(self.fold)
        self.net = net

    def forward(self, x):
        return self.net(torch.cat([self.gs(x[:, :self.fold]), x[:, self.fold:]], dim=1))


def insert_gsm(backbone, stages=("s3", "s4"), n_div=4):
    """Wrap conv1 of every block in the given timm RegNet stages. Returns the GSM modules (to set T)."""
    mods = []
    for name in stages:
        for block in getattr(backbone, name).children():
            block.conv1 = GatedShift(block.conv1, n_div)
            mods.append(block.conv1.gs)
    return mods
