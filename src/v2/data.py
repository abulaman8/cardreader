"""v2 datasets: decoded uint8 frame windows (T, 3, H, W) + confidence / displacement / clip targets."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..dataset import gaussian_targets
from .common import card_times, clip_path, half_prefix
from .frames import JpegSeq


def displacement_targets(T, card_frames, R):
    """d_t = (nearest card - t) / R, supervised only where |card - t| <= R."""
    tgt, mask = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    if len(card_frames) == 0:
        return tgt, mask
    t = np.arange(T, dtype=np.float32)[:, None]
    d = np.asarray(card_frames, dtype=np.float32)[None, :] - t
    j = np.abs(d).argmin(1)
    near = d[np.arange(T), j]
    m = np.abs(near) <= R
    tgt[m], mask[m] = near[m] / R, 1.0
    return tgt, mask


def make_targets(T, rel, cfg):
    tc = cfg["targets"]
    rel = np.asarray(rel, dtype=np.float32)
    y = gaussian_targets(T, rel, tc["sigma"])
    d, m = displacement_targets(T, rel, tc["disp_radius"])
    clip = float(np.any((rel >= 0) & (rel < T)))
    return (torch.from_numpy(y), torch.from_numpy(d), torch.from_numpy(m), torch.tensor(clip, dtype=torch.float32))


def augment(x, rng, aug):
    """x uint8 (T, 3, H, W) -> uint8. One random transform per clip (same for every frame)."""
    T, _, H, W = x.shape
    f = x.float().div_(255.0)
    lo, hi = aug["crop_scale"]
    s = rng.uniform(lo, hi)
    if s < 0.999:
        ch, cw = int(round(H * s)), int(round(W * s))
        y0, x0 = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
        f = F.interpolate(f[:, :, y0:y0 + ch, x0:x0 + cw], size=(H, W), mode="bilinear", align_corners=False)
    if rng.random() < aug["hflip"]:
        f = f.flip(-1)
    j = aug["color_jitter"]
    if j > 0:
        f = f * (1 + rng.uniform(-j, j))
        mean = f.mean(dim=(1, 2, 3), keepdim=True)
        f = (f - mean) * (1 + rng.uniform(-j, j)) + mean
        gray = f.mean(dim=1, keepdim=True)
        f = (f - gray) * (1 + rng.uniform(-j, j)) + gray
    return f.clamp_(0, 1).mul_(255).round_().to(torch.uint8)


class TrainClipsV2(Dataset):
    """Positives (all) + rule-based negatives + mined hard negatives (capped share), resampled per epoch."""

    def __init__(self, cfg, index, mined_specs=()):
        d = cfg["data"]
        self.cfg, self.T, self.m, self.L, self.seed = cfg, d["window"], d["margin"], d["cache_len"], d["seed"]
        self.aug = cfg["augment"]
        self.pos = [(clip_path(cfg, "pos", c["id"]), c["anchor"], c["cards"]) for c in index["train_pos"]]
        self.neg = [clip_path(cfg, "neg", c["id"]) for c in index["train_neg"]]
        self.mined = [clip_path(cfg, "mined", c["id"]) for c in mined_specs]
        for name in ("pos", "neg", "mined"):
            lst = getattr(self, name)
            keep = [x for x in lst if os.path.exists(x[0] if isinstance(x, tuple) else x)]
            if len(keep) < len(lst):
                print(f"[warn] {name}: {len(lst) - len(keep)}/{len(lst)} cached clips missing")
            setattr(self, name, keep)
        if not self.pos or not self.neg:
            raise RuntimeError("no cached training clips; run `python -m src.v2.frames clips` first")
        self.resample(0)

    def resample(self, epoch):
        d = self.cfg["data"]
        rng = np.random.default_rng((self.seed, epoch))
        n_neg = int(round(d["neg_per_pos"] * len(self.pos)))
        n_mined = min(len(self.mined), int(round(d["mined_frac"] * n_neg)))
        n_rule = min(len(self.neg), n_neg - n_mined)
        items = [(0, i) for i in range(len(self.pos))]
        items += [(1, int(i)) for i in rng.choice(len(self.neg), n_rule, replace=False)]
        if n_mined:
            items += [(2, int(i)) for i in rng.choice(len(self.mined), n_mined, replace=False)]
        self.items, self.epoch = items, epoch
        self.counts = {"pos": len(self.pos), "neg": n_rule, "mined": n_mined}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, j = self.items[i]
        rng = np.random.default_rng((self.seed, self.epoch, i, 11))
        T, L, m = self.T, self.L, self.m
        if kind == 0:
            path, anchor, cards = self.pos[j]
            p = int(rng.integers(m, T - m))
            s = int(np.clip(anchor - p, 0, L - T))
        else:
            path, cards = (self.neg[j] if kind == 1 else self.mined[j]), []
            s = int(rng.integers(0, L - T + 1))
        x = augment(JpegSeq.open_clip(path).decode(s, T), rng, self.aug)
        rel = np.asarray(cards, dtype=np.float32) - s
        return (x, *make_targets(T, rel, self.cfg))


class ValClipsV2(Dataset):
    """Deterministic val clips sliced from cached val halves (clip-level metrics during training)."""

    def __init__(self, cfg, index, split="val"):
        d = cfg["data"]
        self.cfg, self.T, fps, m = cfg, d["window"], d["fps"], d["margin"]
        self.prefix, self.cards, self.items = {}, {}, []
        missing = set()
        specs = index["eval"][split]
        for kind in ("pos", "neg"):
            for c in specs[kind]:
                key = c["key"]
                if key not in self.prefix:
                    pre = half_prefix(cfg, split, key)
                    if not os.path.exists(pre + ".off.npy"):
                        missing.add(key)
                        continue
                    self.prefix[key] = pre
                    self.cards[key] = np.array([round(t * fps) for t in card_times(index, key, True)], dtype=np.float32)
                elif key in missing:
                    continue
                N = len(np.load(self.prefix[key] + ".off.npy")) - 1
                c0 = round(c["t"] * fps)
                p = m + int(round(c["u"] * (self.T - 1 - 2 * m))) if kind == "pos" else self.T // 2
                self.items.append((key, int(np.clip(c0 - p, 0, max(N - self.T, 0)))))
        if missing:
            print(f"[warn] {split}: {len(missing)} halves not cached; their clips are skipped")
        self._open = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, s = self.items[i]
        if key not in self._open:
            self._open[key] = JpegSeq.open_half(self.prefix[key])
        x = self._open[key].decode(s, self.T)
        return (x, *make_targets(self.T, self.cards[key] - s, self.cfg))
