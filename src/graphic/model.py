"""Stage-2 Post-Event Broadcast Card Graphic Detector (GraphicNet).

Processes K=18 post-event broadcast overlay strips (B, 18, 3, 96, 256) uint8
across [t + 2.0s, t + 27.5s], a pre-event reference strip (B, 1, 3, 96, 256) uint8
at t - 3.0s, and stationary pop-up graphic features g_aux (B, 18, 8) using:
  1. RegNetY-400MF backbone with Temporal Shift Modules (TSM) + a zero-init
     Pre-Event Pop-Up Difference Stem Adapter (diff_stem) that highlights pixels
     that newly appeared relative to t - 3.0s (suppressing permanent team badges).
  2. 4-head Spatial Attention Pooling over the (top-scorebug + lower-third) strip.
  3. 16-d stationary pop-up badge & substitution-veto feature adapter.
  4. 3-layer Pre-LN Temporal Transformer + Multiple-Instance Learning (MIL)
     contiguous 2-3 step burst pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Block
from src.v2.model import MEAN, STD, build_backbone


def compute_popup_features(f_rgb, r_rgb, att_map, g_aux):
    """Build 16-d per-step stationary pop-up graphic feature vector.
    f_rgb:   (B, K, 3, H, W) float in [0, 1]
    r_rgb:   (B, 1, 3, H, W) float in [0, 1]
    att_map: (B*K, heads, Hf, Wf) spatial attention weights
    g_aux:   (B, K, 8) float [y_peak, y_area, r_peak, r_area, g_peak, top_mask, bot_mask, global_diff]
    Returns: (B, K, 16) float
    """
    B, K, _, H, W = f_rgb.shape
    diff_mag = (f_rgb - r_rgb).abs().mean(dim=2)  # (B, K, H, W)
    popup_gate = (diff_mag > 0.075).float()

    r, g, b = f_rgb[:, :, 0], f_rgb[:, :, 1], f_rgb[:, :, 2]
    yel = torch.clamp(torch.minimum(r, g) - 1.35 * b, min=0.0) * ((r > 0.62) & (g > 0.52) & (b < 0.38)).float() * popup_gate
    red = torch.clamp(r - 1.45 * torch.maximum(g, b), min=0.0) * ((r > 0.62) & (g < 0.32) & (b < 0.32)).float() * popup_gate

    att_mean = att_map.mean(dim=1, keepdim=True)  # (B*K, 1, Hf, Wf)
    att_up = F.interpolate(att_mean, size=(H, W), mode="bilinear", align_corners=False).view(B, K, H, W)
    att_y = (yel * att_up).flatten(2).sum(dim=-1)
    att_r = (red * att_up).flatten(2).sum(dim=-1)

    g0, g1, g2, g3, g4, g5, g6, g7 = [g_aux[..., i] for i in range(8)]
    # Baseline-subtracted temporal appearance spikes within the 18-step window
    y_spike = g0 - g0.amin(dim=1, keepdim=True)
    r_spike = g2 - g2.amin(dim=1, keepdim=True)
    g_spike = g4 - g4.amin(dim=1, keepdim=True)

    # Compact card icon gates (card icon is < 3.5% of strip area and has no green substitution arrow)
    no_sub = torch.clamp(1.0 - 4.0 * g4, min=0.0, max=1.0)
    compact_y = g0 * ((g1 > 0.0004) & (g1 < 0.038)).float() * no_sub
    compact_r = g2 * ((g3 > 0.0004) & (g3 < 0.038)).float() * no_sub
    popup_ratio = (g5 + g6) / (g7 + 0.01)

    extra8 = torch.stack([y_spike, r_spike, g_spike, compact_y, compact_r, popup_ratio, att_y, att_r], dim=-1)
    return torch.cat([g_aux, extra8], dim=-1)


class BroadcastGraphicVerifier(nn.Module):
    def __init__(
        self,
        backbone="regnet_y_400mf",
        pretrained=True,
        tsm_fold_div=8,
        attn_heads=4,
        d_model=256,
        layers=3,
        heads=4,
        ffn=768,
        dropout=0.25,
        attn_dropout=0.1,
        max_len=24,
    ):
        super().__init__()
        self.backbone, self.shifts, C = build_backbone(backbone, pretrained, tsm_fold_div)
        self.backbone.fc = nn.Identity()
        self.attn_heads = attn_heads
        self.max_len = max_len

        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

        # Zero-initialized Pre-Event Pop-Up Difference Stem Adapter (operates on f_rgb - r_rgb)
        self.diff_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.diff_stem[-1].weight)
        nn.init.zeros_(self.diff_stem[-1].bias)

        self.attn_logits = nn.Conv2d(C, attn_heads, kernel_size=1)
        nn.init.zeros_(self.attn_logits.weight)
        nn.init.zeros_(self.attn_logits.bias)

        self.pool_proj = nn.Sequential(
            nn.Linear(C * attn_heads, d_model),
            nn.LayerNorm(d_model),
        )

        feat_dim = 16
        self.gaux_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
        )
        nn.init.zeros_(self.gaux_proj[-1].weight)
        nn.init.zeros_(self.gaux_proj[-1].bias)

        # Direct physical-feature MLP branch for step-level badge scoring
        self.phys_step_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.in_drop = nn.Dropout(0.15)

        self.blocks = nn.ModuleList(
            Block(d_model, heads, ffn, dropout, attn_dropout) for _ in range(layers)
        )
        self.ln_f = nn.LayerNorm(d_model)

        self.step_head = nn.Linear(d_model, 1)
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
        print(f"Initialized BroadcastGraphicVerifier with {loaded} tensors from {ckpt_path}", flush=True)

    def backbone_params(self):
        return list(self.backbone.parameters())

    def forward(self, strips_u8, ref_u8, g_aux):
        """
        strips_u8: (B, K=18, 3, 96, 256) uint8 post-event broadcast overlay strips
        ref_u8:    (B, 1, 3, 96, 256) uint8 pre-event reference strip at t - 3.0s
        g_aux:     (B, K=18, 8) float stationary pop-up graphic features
        Returns:
          step_logits: (B, K) per-step graphic confidence logits
          out_logit:   (B,) combined MIL contiguous-burst + Transformer [CLS] logit
        """
        B, K = strips_u8.shape[:2]
        for s in self.shifts:
            s.n_frames = K

        f_rgb = strips_u8.float().div_(255.0)
        r_rgb = ref_u8.float().div_(255.0)
        d_rgb = (f_rgb - r_rgb).flatten(0, 1)  # (B*K, 3, 96, 256)

        f_norm = (f_rgb.flatten(0, 1) - self.mean) / self.std
        f = self.backbone.stem(f_norm) + self.diff_stem(d_rgb)
        f = self.backbone.trunk_output(f)

        N, C, Hf, Wf = f.shape
        a = self.attn_logits(f).flatten(2).float().softmax(-1)  # (N, heads, Hf*Wf)
        pooled = torch.einsum("nhs,ncs->nhc", a.to(f.dtype), f.flatten(2)).flatten(1)

        feats16 = compute_popup_features(
            f_rgb, r_rgb, a.view(N, self.attn_heads, Hf, Wf), g_aux.float()
        ).to(pooled.dtype)  # (B, K, 16)

        z = (self.pool_proj(pooled).view(B, K, -1) + self.gaux_proj(feats16))
        h = torch.cat([self.cls.to(z.dtype).expand(B, -1, -1), z], dim=1) + self.pos[:, :K + 1].to(z.dtype)
        h = self.in_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        h_cls = h[:, 0]
        h_steps = h[:, 1:]

        step_logits = self.step_head(h_steps).squeeze(-1) + 0.5 * self.phys_step_head(feats16).squeeze(-1)
        cls_logit = self.clip_head(h_cls).squeeze(-1)

        # Contiguous 2-step (3.0s) and 3-step (4.5s) burst MIL pooling
        burst2 = F.avg_pool1d(step_logits.unsqueeze(1), kernel_size=2, stride=1).squeeze(1).amax(dim=1)
        burst3 = F.avg_pool1d(step_logits.unsqueeze(1), kernel_size=3, stride=1).squeeze(1).amax(dim=1)
        mil_logit = 0.5 * burst2 + 0.5 * burst3

        out_logit = 0.5 * cls_logit + 0.5 * mil_logit
        return step_logits, out_logit
