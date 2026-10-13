"""CardNetV2: RegNetY-400MF (+ temporal shift) on full frames -> learned spatial attention pooling
-> temporal transformer -> confidence / displacement / clip heads.

Input: uint8 (B, T, 3, H, W). Normalisation happens inside the model.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.checkpoint import checkpoint

from ..model import Block

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TemporalShift(nn.Module):
    """TSM: shift 1/fold_div channels from t+1 and 1/fold_div from t-1. No parameters, no FLOPs."""

    def __init__(self, fold_div=8):
        super().__init__()
        self.fold_div, self.n_frames = fold_div, 1

    def forward(self, x):
        T = self.n_frames
        if T <= 1:
            return x
        nt, c, h, w = x.shape
        x = x.view(nt // T, T, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class Shifted(nn.Module):
    def __init__(self, shift, f):
        super().__init__()
        self.shift, self.f = shift, f

    def forward(self, x):
        return self.f(self.shift(x))


def build_backbone(name, pretrained, fold_div):
    weights = tvm.get_model_weights(name).DEFAULT if pretrained else None
    net = getattr(tvm, name)(weights=weights)
    shifts = []
    for stage in net.trunk_output.children():
        for block in stage.children():
            s = TemporalShift(fold_div)
            block.f = Shifted(s, block.f)         # shift inside the residual branch
            shifts.append(s)
    return net, shifts, net.fc.in_features


class CardNetV2(nn.Module):
    def __init__(self, backbone="regnet_y_400mf", pretrained=True, tsm_fold_div=8, attn_heads=4, d_model=256,
                 layers=4, heads=4, ffn=1024, dropout=0.2, attn_dropout=0.1, input_dropout=0.1, max_len=45,
                 grad_checkpoint=False):
        super().__init__()
        self.backbone, self.shifts, C = build_backbone(backbone, pretrained, tsm_fold_div)
        self.backbone.fc = nn.Identity()
        self.grad_checkpoint, self.max_len, self.attn_heads = grad_checkpoint, max_len, attn_heads
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.attn_logits = nn.Conv2d(C, attn_heads, kernel_size=1)          # learned "where to look" per location
        self.pool_proj = nn.Sequential(nn.Linear(C * attn_heads, d_model), nn.LayerNorm(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        self.in_drop = nn.Dropout(input_dropout)
        self.blocks = nn.ModuleList(Block(d_model, heads, ffn, dropout, attn_dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.conf_head = nn.Linear(d_model, 1)
        self.disp_head = nn.Linear(d_model, 1)
        self.clip_head = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.attn_logits.weight)                              # start as uniform (mean) pooling
        nn.init.zeros_(self.attn_logits.bias)

    def backbone_params(self):
        return list(self.backbone.parameters())

    def frame_features(self, x):
        B, T = x.shape[:2]
        for s in self.shifts:
            s.n_frames = T
        f = x.flatten(0, 1).float().div(255.0)
        f = (f - self.mean) / self.std
        f = self.backbone.stem(f)
        if self.grad_checkpoint and self.training:
            for stage in self.backbone.trunk_output.children():
                f = checkpoint(stage, f, use_reentrant=False)
        else:
            f = self.backbone.trunk_output(f)
        N, C, Hf, Wf = f.shape
        a = self.attn_logits(f).flatten(2).float().softmax(-1)               # (N, heads, HW)
        pooled = torch.einsum("nhs,ncs->nhc", a.to(f.dtype), f.flatten(2)).flatten(1)
        z = self.pool_proj(pooled).view(B, T, -1)
        return z, a.view(B, T, self.attn_heads, Hf, Wf)

    def forward(self, x, return_attn=False):
        B, T = x.shape[:2]
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len={self.max_len}")
        z, a = self.frame_features(x)
        h = torch.cat([self.cls.to(z.dtype).expand(B, -1, -1), z], dim=1) + self.pos[:, :T + 1].to(z.dtype)
        h = self.in_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        conf = self.conf_head(h[:, 1:]).squeeze(-1)
        disp = self.disp_head(h[:, 1:]).squeeze(-1)
        clip = self.clip_head(h[:, 0]).squeeze(-1)
        return (conf, disp, clip, a) if return_attn else (conf, disp, clip)


def build_model_v2(cfg, pretrained=None):
    m = cfg["model"]
    return CardNetV2(backbone=m["backbone"], pretrained=m["pretrained"] if pretrained is None else pretrained,
                     tsm_fold_div=m["tsm_fold_div"], attn_heads=m["attn_heads"], d_model=m["d_model"],
                     layers=m["layers"], heads=m["heads"], ffn=m["ffn"], dropout=m["dropout"],
                     attn_dropout=m["attn_dropout"], input_dropout=m["input_dropout"],
                     max_len=cfg["data"]["cache_len"], grad_checkpoint=m["grad_checkpoint"])


def load_model_v2(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model_v2(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["config"], ckpt
