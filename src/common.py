"""Small shared helpers: index access, feature paths, device selection."""
import json
import os

import torch

HALVES = (1, 2)


def load_index(cfg):
    with open(cfg["paths"]["index"]) as f:
        return json.load(f)


def half_key(game, half):
    return f"{game}|{half}"


def split_key(key):
    game, half = key.rsplit("|", 1)
    return game, int(half)


def split_keys(index, split):
    """All half keys (game|half) of a split that exist in the index."""
    return [half_key(g, h) for g in index["splits"][split] for h in HALVES
            if half_key(g, h) in index["halves"]]


def half_feat_path(cfg, split, key):
    game, half = split_key(key)
    return os.path.join(cfg["paths"]["features"], "eval", split, game, f"half{half}.npy")


def train_feat_path(cfg, kind, clip_id):
    return os.path.join(cfg["paths"]["features"], "train", kind, f"{clip_id}.npy")


def card_times(index, key, visible=True):
    """Card times (s) in a half. visible=True -> 'visible' cards, False -> 'not shown' cards."""
    return [c["t"] for c in index["halves"][key]["cards"] if (c["visibility"] == "visible") == visible]


def get_device(name="cuda"):
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)
