"""Stage-2 Referee RoI Tubelet Verifier Model.

Processes (B, T=25, 3, 128, 128) uint8 referee tubelets + per-frame auxiliary
signals (B, T, 4) [v2_prob, v2_disp, det_score, card_sal] using a RegNetY-400MF
backbone with Temporal Shift Modules (TSM), 4-head Spatial Attention Pooling,
explicit yellow/red card chromaticity pooling, and a 3-layer Pre-LN Temporal
Transformer to distinguish true card bookings from empty-hand referee gestures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Block
from src.v2.model import MEAN, STD, build_backbone


def compute_chromaticity_stats(f_rgb, att_map):
    """Extract per-frame yellow/red card chromaticity statistics on (N, 3, H, W) float in [0, 1].
    att_map: (N, heads, Hf, Wf) spatial attention weights.
    Returns (N, 8) chromaticity feature vector.
    """
    N, _, H, W = f_rgb.shape
    r, g, b = f_rgb[:, 0], f_rgb[:, 1], f_rgb[:, 2]

    # Strict yellow & red chromaticity masks
    yellow = torch.clamp(torch.minimum(r, g) - 1.25 * b, min=0.0) * ((r > 0.58) & (g > 0.48)).float()
    red = torch.clamp(r - 1.35 * torch.maximum(g, b), min=0.0) * (r > 0.58).float()

    # Upper 65% of crop (where raised hand/card appears)
    u_h = int(H * 0.65)
    y_up = yellow[:, :u_h, :]
    r_up = red[:, :u_h, :]

    # Global upper mean & max-pool (4x4 patch max to capture a ~10x16 px card)
    y_patches = F.avg_pool2d(y_up[:, None], kernel_size=8, stride=4)
    r_patches = F.avg_pool2d(r_up[:, None], kernel_size=8, stride=4)

    y_max = y_patches.flatten(1).max(dim=-1)[0]
    y_mean = y_up.flatten(1).mean(dim=-1)
    r_max = r_patches.flatten(1).max(dim=-1)[0]
    r_mean = r_up.flatten(1).mean(dim=-1)

    # Attention-weighted yellow & red saliency
    att_mean = att_map.mean(dim=1, keepdim=True)  # (N, 1, Hf, Wf)
    att_up = F.interpolate(att_mean, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    y_att = (yellow * att_up).flatten(1).sum(dim=-1)
    r_att = (red * att_up).flatten(1).sum(dim=-1)

    # Top-5 pixel average
    y_top = torch.topk(yellow.flatten(1), k=min(25, H * W), dim=-1)[0].mean(dim=-1)
    r_top = torch.topk(red.flatten(1), k=min(25, H * W), dim=-1)[0].mean(dim=-1)

    return torch.stack([y_max, y_mean, y_att, y_top, r_max, r_mean, r_att, r_top], dim=-1)


class RefereeRoIVerifier(nn.Module):
    def __init__(
        self,
        backbone="regnet_y_400mf",
        pretrained=True,
        tsm_fold_div=8,
        attn_heads=4,
        d_model=256,
        layers=4,
        heads=4,
        ffn=1024,
        dropout=0.2,
        attn_dropout=0.1,
        max_len=45,
    ):
        super().__init__()
        self.backbone, self.shifts, C = build_backbone(backbone, pretrained, tsm_fold_div)
        self.backbone.fc = nn.Identity()
        self.attn_heads = attn_heads
        self.max_len = max_len

        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

        self.attn_logits = nn.Conv2d(C, attn_heads, kernel_size=1)
        nn.init.zeros_(self.attn_logits.weight)
        nn.init.zeros_(self.attn_logits.bias)

        self.pool_proj = nn.Sequential(
            nn.Linear(C * attn_heads, d_model),
            nn.LayerNorm(d_model),
        )

        # Zero-initialized residual adapter for 12-d chromaticity + temporal spike + detector features
        extra_dim = 12
        self.chrom_proj = nn.Sequential(
            nn.LayerNorm(extra_dim),
            nn.Linear(extra_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
        )
        nn.init.zeros_(self.chrom_proj[-1].weight)
        nn.init.zeros_(self.chrom_proj[-1].bias)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.in_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            Block(d_model, heads, ffn, dropout, attn_dropout) for _ in range(layers)
        )
        self.ln_f = nn.LayerNorm(d_model)

        self.frame_head = nn.Linear(d_model, 1)
        self.clip_head = nn.Linear(d_model, 1)

    def init_from_v2(self, ckpt_path="runs/v2_r2/best.pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
        own = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            elif k == "conf_head.weight" and "frame_head.weight" in own:
                own["frame_head.weight"].copy_(v)
                loaded += 1
            elif k == "conf_head.bias" and "frame_head.bias" in own:
                own["frame_head.bias"].copy_(v)
                loaded += 1
        print(f"Initialized RefereeRoIVerifier with {loaded} tensors from {ckpt_path}", flush=True)

    def backbone_params(self):
        return list(self.backbone.parameters())

    def forward(self, x_u8, aux=None):
        """
        x_u8: (B, T, 3, 128, 128) uint8 referee tubelet
        aux:  (B, T, 4) float [v2_prob, v2_disp, det_score, card_sal] or None
        Returns:
          frame_logits: (B, T)
          clip_logit:   (B,)
        """
        B, T = x_u8.shape[:2]
        for s in self.shifts:
            s.n_frames = T

        f_rgb = x_u8.flatten(0, 1).float().div_(255.0)
        f = (f_rgb - self.mean) / self.std
        f = self.backbone.stem(f)
        f = self.backbone.trunk_output(f)

        N, C, Hf, Wf = f.shape
        a = self.attn_logits(f).flatten(2).float().softmax(-1)  # (N, heads, Hf*Wf)
        pooled = torch.einsum("nhs,ncs->nhc", a.to(f.dtype), f.flatten(2)).flatten(1)  # (N, C*heads)

        chrom = compute_chromaticity_stats(f_rgb, a.view(N, self.attn_heads, Hf, Wf)).view(B, T, 8)
        y_spike = chrom[:, :, 0] - chrom[:, :, 0].amin(dim=1, keepdim=True)
        r_spike = chrom[:, :, 4] - chrom[:, :, 4].amin(dim=1, keepdim=True)
        spikes = torch.stack([y_spike, r_spike], dim=-1).reshape(N, 2)

        if aux is None:
            det_sal = torch.zeros((N, 2), device=x_u8.device, dtype=pooled.dtype)
        else:
            det_sal = aux[..., 2:4].reshape(N, 2).to(device=x_u8.device, dtype=pooled.dtype)

        extra = torch.cat([chrom.reshape(N, 8).to(pooled.dtype), spikes.to(pooled.dtype), det_sal], dim=-1)
        z = (self.pool_proj(pooled) + self.chrom_proj(extra)).view(B, T, -1)

        h = torch.cat([self.cls.to(z.dtype).expand(B, -1, -1), z], dim=1) + self.pos[:, :T + 1].to(z.dtype)
        h = self.in_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        h_cls = h[:, 0]     # (B, D)
        h_frames = h[:, 1:] # (B, T, D)
        frame_logits = self.frame_head(h_frames).squeeze(-1)  # (B, T)
        clip_logit = self.clip_head(h_cls).squeeze(-1)        # (B,)
        return frame_logits, clip_logit
