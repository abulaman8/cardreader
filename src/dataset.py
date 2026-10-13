"""Datasets over cached features.

TrainClips: train-game clips only; positives cropped with jitter so the card lands in [margin, T-1-margin];
            negatives resampled from the pool every epoch.
EvalClips:  held-out val/test clips sliced from full-half features (deterministic).
Targets:    per-frame Gaussian (sigma frames, zero beyond 3 sigma, max over cards) + clip label.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import card_times, half_feat_path, train_feat_path


def gaussian_targets(T, card_frames, sigma):
    y = np.zeros(T, dtype=np.float32)
    if len(card_frames) == 0:
        return y
    d = np.arange(T, dtype=np.float32)[:, None] - np.asarray(card_frames, dtype=np.float32)[None, :]
    g = np.exp(-d ** 2 / (2 * sigma ** 2))
    g[np.abs(d) > 3 * sigma] = 0.0
    return g.max(1).astype(np.float32)


def _clip_label(rel, T):
    rel = np.asarray(rel)
    return float(np.any((rel >= 0) & (rel < T)))


class TrainClips(Dataset):
    def __init__(self, cfg, index):
        d = cfg["data"]
        self.T, self.m, self.L = d["window"], d["margin"], d["cache_len"]
        self.sigma = cfg["targets"]["sigma"]
        self.flip_p = cfg["train"]["flip_p"]
        self.neg_per_pos = d["neg_per_pos"]
        self.seed = d["seed"]
        self.pos = [(train_feat_path(cfg, "pos", c["id"]), c["anchor"], c["cards"]) for c in index["train_pos"]]
        self.neg = [train_feat_path(cfg, "neg", c["id"]) for c in index["train_neg"]]
        n_pos, n_neg = len(self.pos), len(self.neg)
        self.pos = [p for p in self.pos if os.path.exists(p[0])]
        self.neg = [p for p in self.neg if os.path.exists(p)]
        if len(self.pos) < n_pos or len(self.neg) < n_neg:
            print(f"[warn] missing features: pos {n_pos - len(self.pos)}/{n_pos}, neg {n_neg - len(self.neg)}/{n_neg}")
        if not self.pos or not self.neg:
            raise RuntimeError("no training features found; run `python -m src.extract clips` first")
        self.resample(0)

    def resample(self, epoch):
        rng = np.random.default_rng((self.seed, epoch))
        k = min(len(self.neg), self.neg_per_pos * len(self.pos))
        neg_idx = rng.choice(len(self.neg), k, replace=False)
        self.items = [(0, i) for i in range(len(self.pos))] + [(1, int(i)) for i in neg_idx]
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        is_neg, j = self.items[i]
        rng = np.random.default_rng((self.seed, self.epoch, i, 7))
        T, L, m = self.T, self.L, self.m
        if is_neg:
            path, cards = self.neg[j], []
            s = int(rng.integers(0, L - T + 1))
        else:
            path, anchor, cards = self.pos[j]
            p = int(rng.integers(m, T - m))              # card position in window: [m, T-1-m]
            s = int(np.clip(anchor - p, 0, L - T))
        arr = np.load(path, mmap_mode="r")               # (2, L, D): [orig, flip]
        v = 1 if rng.random() < self.flip_p else 0
        x = np.asarray(arr[v, s:s + T], dtype=np.float32)
        rel = np.asarray(cards, dtype=np.float32) - s
        y = gaussian_targets(T, rel, self.sigma)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(_clip_label(rel, T), dtype=torch.float32)


class EvalClips(Dataset):
    def __init__(self, cfg, index, split):
        d = cfg["data"]
        self.T, self.m, fps = d["window"], d["margin"], d["fps"]
        self.sigma = cfg["targets"]["sigma"]
        self.feats, self.cards, self.items = {}, {}, []
        specs = index["eval"][split]
        missing = set()

        def feats_for(key):
            if key not in self.feats:
                path = half_feat_path(cfg, split, key)
                if not os.path.exists(path):
                    missing.add(key)
                    return None
                self.feats[key] = np.load(path, mmap_mode="r")
                self.cards[key] = np.array([round(t * fps) for t in card_times(index, key, True)], dtype=np.float32)
            return self.feats[key]

        T, m = self.T, self.m
        for c in specs["pos"]:
            f = feats_for(c["key"])
            if f is None or len(f) < T:
                continue
            p = m + int(round(c["u"] * (T - 1 - 2 * m)))
            self.items.append((c["key"], int(np.clip(round(c["t"] * fps) - p, 0, len(f) - T))))
        for c in specs["neg"]:
            f = feats_for(c["key"])
            if f is None or len(f) < T:
                continue
            self.items.append((c["key"], int(np.clip(round(c["t"] * fps) - T // 2, 0, len(f) - T))))
        if missing:
            print(f"[warn] {split}: {len(missing)} halves have no features; their clips are skipped")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, s = self.items[i]
        x = np.asarray(self.feats[key][s:s + self.T], dtype=np.float32)
        rel = self.cards[key] - s
        y = gaussian_targets(self.T, rel, self.sigma)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(_clip_label(rel, self.T), dtype=torch.float32)
