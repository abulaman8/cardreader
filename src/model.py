"""CardTransformer: small pre-LN temporal transformer over frozen per-frame features.

Heads: per-frame confidence (T logits, Gaussian-smoothed targets) + clip logit on a CLS token.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, heads, ffn, dropout, attn_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ff(self.ln2(x)))


class CardTransformer(nn.Module):
    def __init__(self, in_dim=768, d_model=256, layers=4, heads=4, ffn=1024, dropout=0.2, attn_dropout=0.1,
                 input_dropout=0.1, token_drop=0.1, feat_noise=0.05, max_len=90):
        super().__init__()
        self.token_drop, self.feat_noise, self.max_len = token_drop, feat_noise, max_len
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.in_drop = nn.Dropout(input_dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        self.blocks = nn.ModuleList(Block(d_model, heads, ffn, dropout, attn_dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.frame_head = nn.Linear(d_model, 1)
        self.clip_head = nn.Linear(d_model, 1)
        for p in (self.cls, self.mask_token, self.pos):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x):
        """x: (B, T, in_dim) -> frame_logits (B, T), clip_logits (B,)"""
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        h = self.proj(x)
        if self.training and self.feat_noise > 0:
            h = h + torch.randn_like(h) * self.feat_noise
        if self.training and self.token_drop > 0:
            m = torch.rand(B, T, 1, device=h.device) < self.token_drop
            h = torch.where(m, self.mask_token.to(h.dtype).expand(B, T, -1), h)
        h = torch.cat([self.cls.to(h.dtype).expand(B, -1, -1), h], dim=1) + self.pos[:, :T + 1].to(h.dtype)
        h = self.in_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.frame_head(h[:, 1:]).squeeze(-1), self.clip_head(h[:, 0]).squeeze(-1)


def build_model(cfg):
    m = cfg["model"]
    return CardTransformer(in_dim=cfg["encoder"]["dim"], d_model=m["d_model"], layers=m["layers"], heads=m["heads"],
                           ffn=m["ffn"], dropout=m["dropout"], attn_dropout=m["attn_dropout"],
                           input_dropout=m["input_dropout"], token_drop=m["token_drop"], feat_noise=m["feat_noise"],
                           max_len=cfg["data"]["cache_len"])


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["config"], ckpt
