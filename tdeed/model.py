"""T-DEED: RegNetY-200MF + GSM (per-frame) -> SGP encoder / SGP-Mixer decoder -> class + displacement heads.

Input: (B, T, 3, H, W) uint8, or float in [0, 1] (augmented). ImageNet normalisation happens inside.
Output: logits (B, T, 2) [background, card], displacement (B, T) in frames (event at t - disp).
Temporal modules follow github.com/arturxe2/T-DEED model/modules.py (SGPBlock, SGPMixer, EDSGPMIXERLayers).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .gsm import insert_gsm

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over C for (B, C, T)."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, c, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        r = x - mu
        var = (r * r).mean(dim=1, keepdim=True)
        return r / torch.sqrt(var + self.eps) * self.weight + self.bias


def _dw(c, k):
    return nn.Conv1d(c, c, k, stride=1, padding=k // 2, groups=c)


def _up_size(ks, r):
    u = round((ks + 1) * r)
    return u + 1 if u % 2 == 0 else u


def _init(convs, std):
    for m in convs:
        nn.init.normal_(m.weight, 0.0, std)
        nn.init.zeros_(m.bias)


class SGPBlock(nn.Module):
    """Scalable-Granularity Perception (TriDet), T-DEED 'normal' mode."""

    def __init__(self, c, ks=9, r=4, init_std=0.1):
        super().__init__()
        assert ks % 2 == 1
        up = _up_size(ks, r)
        self.ln = ChannelLayerNorm(c)
        self.gn = nn.GroupNorm(16, c)
        self.psi, self.convw, self.convkw = _dw(c, ks), _dw(c, ks), _dw(c, up)
        self.fc, self.global_fc = _dw(c, 1), _dw(c, 1)
        self.mlp = nn.Sequential(nn.Conv1d(c, 4 * c, 1), nn.GELU(), nn.Conv1d(4 * c, c, 1))
        _init([self.psi, self.convw, self.convkw, self.fc, self.global_fc], init_std)

    def forward(self, x):                                       # (B, C, T)
        out = self.ln(x)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = self.fc(out) * phi + (self.convw(out) + self.convkw(out)) * self.psi(out) + out
        out = x + out
        return out + self.mlp(self.gn(out))


class SGPMixer(nn.Module):
    """Decoder fusion: upsampled coarse features x + skip features z, SGP branches on both, concat + 1x1."""

    def __init__(self, c, ks=9, r=4, init_std=0.1):
        super().__init__()
        up = _up_size(ks, r)
        self.ln1, self.ln2 = ChannelLayerNorm(c), ChannelLayerNorm(c)
        self.gn = nn.GroupNorm(16, c)
        self.psi1, self.psi2 = _dw(c, ks), _dw(c, ks)
        self.convw1, self.convkw1 = _dw(c, ks), _dw(c, up)
        self.convw2, self.convkw2 = _dw(c, ks), _dw(c, up)
        self.fc1, self.global_fc1 = _dw(c, 1), _dw(c, 1)
        self.fc2, self.global_fc2 = _dw(c, 1), _dw(c, 1)
        self.concat_fc = nn.Conv1d(6 * c, c, 1)
        self.act = nn.GELU()
        self.mlp = nn.Sequential(nn.Conv1d(c, 4 * c, 1), nn.GELU(), nn.Conv1d(4 * c, c, 1))
        _init([self.psi1, self.psi2, self.convw1, self.convkw1, self.convw2, self.convkw2, self.fc1, self.fc2,
               self.global_fc1, self.global_fc2, self.concat_fc], init_std)

    def forward(self, x, z):
        z = self.ln1(z)
        x = F.interpolate(self.ln2(x), size=z.shape[-1], mode="linear", align_corners=True)
        out1 = (self.convw1(z) + self.convkw1(z)) * self.psi1(z)
        out2 = (self.convw2(x) + self.convkw2(x)) * self.psi2(x)
        out3 = self.fc1(z) * torch.relu(self.global_fc1(z.mean(dim=-1, keepdim=True)))
        out4 = self.fc2(x) * torch.relu(self.global_fc2(x.mean(dim=-1, keepdim=True)))
        out = self.act(self.concat_fc(torch.cat([out1, out2, out3, out4, z, x], dim=1)))
        return out + self.mlp(self.gn(out))


class EDSGPMixer(nn.Module):
    """Encoder (SGP + max-pool /2) x n -> bottleneck SGP -> decoder (SGP-Mixer with skip + SGP) x n."""

    def __init__(self, c, n_layers=2, ks=9, r=4):
        super().__init__()
        self.n = n_layers
        self.sgp = nn.ModuleList(SGPBlock(c, ks, r) for _ in range(2 * n_layers + 1))
        self.mix = nn.ModuleList(SGPMixer(c, ks, r) for _ in range(n_layers))

    def forward(self, x):                                       # (B, T, C)
        x = x.transpose(1, 2)
        skips = []
        for i in range(self.n):
            x = self.sgp[i](x)
            skips.append(x)
            x = F.adaptive_max_pool1d(x, math.ceil(x.shape[-1] / 2))
        x = self.sgp[self.n](x)
        for i in range(self.n):
            x = self.mix[-(i + 1)](x, skips[-(i + 1)])
            x = self.sgp[self.n + i + 1](x)
        return x.transpose(1, 2)


class TDEED(nn.Module):
    def __init__(self, backbone="regnety_002", pretrained=True, shift="gsm", shift_div=4, shift_stages=("s3", "s4"),
                 clip_len=100, d_model=256, n_layers=2, sgp_ks=9, sgp_r=4, head_dropout=0.5, num_classes=1,
                 grad_checkpoint=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.gsms = insert_gsm(self.backbone, tuple(shift_stages), shift_div) if shift == "gsm" else []
        D = self.backbone.num_features
        d = d_model or D
        assert d % 16 == 0, "GroupNorm(16, d) needs d divisible by 16"
        self.clip_len, self.grad_checkpoint = clip_len, grad_checkpoint
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.Identity() if d == D else nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        self.temp_enc = nn.Parameter(torch.randn(clip_len, d) / clip_len)
        self.temporal = EDSGPMixer(d, n_layers, sgp_ks, sgp_r)
        self.cls_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d, num_classes + 1))
        self.disp_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d, 1))

    def backbone_params(self):
        return list(self.backbone.parameters())

    def _features(self, x):                                     # (N, 3, H, W) normalised -> (N, D)
        bb = self.backbone
        if self.grad_checkpoint and self.training:
            x = bb.stem(x)
            for stage in (bb.s1, bb.s2, bb.s3, bb.s4):
                x = checkpoint(stage, x, use_reentrant=True)
            x = bb.final_conv(x)
            return bb.forward_head(x, pre_logits=True)
        return bb(x)

    def forward(self, x):
        B, T = x.shape[:2]
        assert T == self.clip_len, f"clip length {T} != {self.clip_len} (positional encoding is fixed-length)"
        for g in self.gsms:
            g.num_segments = T
        f = x.flatten(0, 1)
        f = f.float().div(255.0) if f.dtype == torch.uint8 else f.float()
        f = (f - self.mean) / self.std
        feat = self.proj(self._features(f).view(B, T, -1)) + self.temp_enc[None]
        feat = self.temporal(feat)
        return self.cls_head(feat), self.disp_head(feat).squeeze(-1)


def vote(prob, disp):
    """T-DEED displacement aggregation: frame t contributes its score to t - round(disp), max per target."""
    B, T = prob.shape
    t = torch.arange(T, device=prob.device).expand(B, T)
    idx = (t - disp.round().long()).clamp(0, T - 1)
    return torch.zeros_like(prob).scatter_reduce(1, idx, prob, reduce="amax", include_self=True)


def build_model(cfg, pretrained=None):
    m, d = cfg["model"], cfg["data"]
    return TDEED(backbone=m["backbone"], pretrained=m["pretrained"] if pretrained is None else pretrained,
                 shift=m["shift"], shift_div=m["shift_div"], shift_stages=m["shift_stages"], clip_len=d["clip_len"],
                 d_model=m["d_model"], n_layers=m["n_layers"], sgp_ks=m["sgp_ks"], sgp_r=m["sgp_r"],
                 head_dropout=m["head_dropout"], grad_checkpoint=m["grad_checkpoint"])


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt
