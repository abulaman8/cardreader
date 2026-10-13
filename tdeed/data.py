"""Datasets for the T-DEED replica. Items carry raw JPEG bytes; decoding happens batched on the GPU
(common.decode_batch), so DataLoader workers are optional (train.num_workers: 0 by default)."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import JpegSeq, eval_prefix, window_path


def make_targets(T, cards, ignores, radius, ign_radius):
    """T-DEED labels. Frames within +-radius of a visible card are 'card' (1) with displacement i - card
    (frames, nearest card wins). Frames within +-ign_radius of a 'not shown' card get zero loss weight,
    unless they are labelled card."""
    cls = np.zeros(T, np.float32)
    disp = np.zeros(T, np.float32)
    best = np.full(T, np.inf, np.float32)
    for c in cards:
        for i in range(max(0, c - radius), min(T, c + radius + 1)):
            if abs(i - c) < best[i]:
                best[i], cls[i], disp[i] = abs(i - c), 1.0, i - c
    mask = np.ones(T, np.float32)
    for g in ignores:
        lo, hi = max(0, g - ign_radius), min(T, g + ign_radius + 1)
        if hi > lo:
            mask[lo:hi] = 0.0
    mask[cls > 0] = 1.0
    return torch.from_numpy(cls), torch.from_numpy(disp), torch.from_numpy(mask)


def collate(batch):
    """-> (list of B lists of T JPEG tensors, cls (B,T), disp (B,T), mask (B,T))."""
    return ([b[0] for b in batch], torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]))


class TrainWindows(Dataset):
    """All positive windows + rule-based negatives (+ capped mined negatives), resampled per epoch.
    A clip_len crop is drawn per item; for positives the anchor card lands anywhere inside the clip."""

    def __init__(self, cfg, index, mined_specs=()):
        d, t = cfg["data"], cfg["targets"]
        self.cfg, self.T, self.seed = cfg, int(d["clip_len"]), d["seed"]
        self.R, self.ign = int(t["radius"]), int(round(d["ignore_radius_s"] * d["fps"]))
        self.groups = {}
        for kind, specs in (("pos", index["train_pos"]), ("neg", index["train_neg"]), ("mined", list(mined_specs))):
            keep = [s for s in specs if os.path.exists(window_path(cfg, kind, s["id"]))]
            if len(keep) < len(specs):
                print(f"[warn] {kind}: {len(specs) - len(keep)}/{len(specs)} windows not extracted, skipped")
            self.groups[kind] = keep
        if not self.groups["pos"] or not self.groups["neg"]:
            raise RuntimeError("no extracted train windows; run `python -m tdeed.extract windows` first")
        self.resample(0)

    def resample(self, epoch):
        d = self.cfg["data"]
        rng = np.random.default_rng((self.seed, epoch))
        n_pos = len(self.groups["pos"])
        n_neg = int(round(d["neg_per_pos"] * n_pos))
        n_mined = min(len(self.groups["mined"]), int(round(d["mined_frac"] * n_neg)))
        n_rule = min(len(self.groups["neg"]), n_neg - n_mined)
        items = [("pos", i) for i in range(n_pos)]
        items += [("neg", int(i)) for i in rng.choice(len(self.groups["neg"]), n_rule, replace=False)]
        if n_mined:
            items += [("mined", int(i)) for i in rng.choice(len(self.groups["mined"]), n_mined, replace=False)]
        self.items, self.epoch = items, epoch
        self.counts = {"pos": n_pos, "neg": n_rule, "mined": n_mined}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, j = self.items[i]
        spec = self.groups[kind][j]
        rng = np.random.default_rng((self.seed, self.epoch, i, 13))
        seq = JpegSeq.open_npz(window_path(self.cfg, kind, spec["id"]))
        n, T = len(seq), self.T
        if kind == "pos":
            a = int(spec["anchor"])
            lo, hi = max(0, a - T + 1), min(a, n - T)
            s = int(rng.integers(lo, hi + 1)) if hi >= lo else int(np.clip(a - T // 2, 0, max(n - T, 0)))
        else:
            s = int(rng.integers(0, max(n - T, 0) + 1))
        cls, disp, mask = make_targets(T, [c - s for c in spec["cards"]], [g - s for g in spec["ignore"]],
                                       self.R, self.ign)
        return seq.raw_range(s, T), cls, disp, mask


class ValClips(Dataset):
    """Deterministic clip_len clips sliced from the existing val half cache (clip-level loss / AUC)."""

    def __init__(self, cfg, index, split="val"):
        d = cfg["data"]
        self.cfg, self.T, fps = cfg, int(d["clip_len"]), d["fps"]
        self.R, self.ign = int(cfg["targets"]["radius"]), int(round(d["ignore_radius_s"] * fps))
        self.prefix, self.cards, self.hidden, self.items = {}, {}, {}, []
        missing = set()
        for c in index["val_clips"]:
            key = c["key"]
            if key not in self.prefix and key not in missing:
                pre = eval_prefix(cfg, split, key)
                if not os.path.exists(pre + ".off.npy"):
                    missing.add(key)
                    continue
                self.prefix[key] = pre
                cards = index["halves"][key]["cards"]
                self.cards[key] = [int(round(x["t"] * fps)) for x in cards if x["visibility"] == "visible"]
                self.hidden[key] = [int(round(x["t"] * fps)) for x in cards if x["visibility"] != "visible"]
            if key in missing:
                continue
            self.items.append((key, int(c["start"])))
        if missing:
            print(f"[warn] {split}: {len(missing)} halves not cached; their clips are skipped")
        self._open = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, s = self.items[i]
        if key not in self._open:
            self._open[key] = JpegSeq.open_half(self.prefix[key])
        seq = self._open[key]
        s = int(min(s, max(len(seq) - self.T, 0)))
        cls, disp, mask = make_targets(self.T, [c - s for c in self.cards[key]], [g - s for g in self.hidden[key]],
                                       self.R, self.ign)
        return seq.raw_range(s, self.T), cls, disp, mask
