"""Shared helpers for the T-DEED replica: config, split/index access, frame-cache paths, JPEG access.

Self-contained: imports nothing from src/. Reads the frozen benchmark split and the existing 5 fps
val/test half cache (frames/eval/<split>/<game>/half{1,2}.{data,off}.npy) read-only.
"""
import json
import os

import numpy as np
import torch
import yaml
from torchvision.io import decode_jpeg

HALVES = (1, 2)
CARD_LABELS = {"Yellow card", "Red card", "Yellow->red card"}
FOUL_LABELS = {"Foul", "Direct free-kick", "Indirect free-kick"}
STOPPAGE_LABELS = {"Substitution", "Offside", "Kick-off", "Throw-in"}
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


# ---------------------------------------------------------------- config
def add_config_args(parser):
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. train.batch_size=2 model.d_model=null")
    return parser


def load_config(path=DEFAULT_CONFIG, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides or ():
        key, val = kv.split("=", 1)
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def get_device(name="cuda"):
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------- split / index
def half_key(game, half):
    return f"{game}|{half}"


def split_key(key):
    game, half = key.rsplit("|", 1)
    return game, int(half)


def keys_for_games(games):
    return [half_key(g, h) for g in games for h in HALVES]


def load_split(cfg):
    with open(cfg["paths"]["split"]) as f:
        return json.load(f)


def load_index(cfg):
    with open(cfg["paths"]["index"]) as f:
        return json.load(f)


def card_times(index, key, visible=True):
    """Card times (s) in a half. visible=True -> 'visible' cards, False -> 'not shown' cards."""
    return [c["t"] for c in index["halves"][key]["cards"] if (c["visibility"] == "visible") == visible]


def gt_dicts(index, keys):
    gts = {k: np.asarray(card_times(index, k, True), dtype=np.float64) for k in keys}
    ign = {k: np.asarray(card_times(index, k, False), dtype=np.float64) for k in keys}
    return gts, ign


# ---------------------------------------------------------------- paths
def eval_prefix(cfg, split, key):
    """Existing full-half cache: <prefix>.data.npy (concatenated JPEG bytes) + <prefix>.off.npy (offsets)."""
    game, half = split_key(key)
    return os.path.join(cfg["paths"]["eval_frames"], split, game, f"half{half}")


def window_path(cfg, kind, wid):
    return os.path.join(cfg["paths"]["train_frames"], kind, f"{wid}.npz")


# ---------------------------------------------------------------- JPEG sequences
class JpegSeq:
    """Random access to a JPEG frame sequence (half cache: memory-mapped; window: .npz)."""

    def __init__(self, data, off):
        self.data, self.off = data, off

    @classmethod
    def open_half(cls, prefix):
        return cls(np.load(prefix + ".data.npy", mmap_mode="r"), np.load(prefix + ".off.npy"))

    @classmethod
    def open_npz(cls, path):
        with np.load(path) as z:
            return cls(z["data"], z["off"])

    def __len__(self):
        return len(self.off) - 1

    def raw(self, i):
        return torch.from_numpy(np.array(self.data[self.off[i]:self.off[i + 1]]))

    def raw_range(self, start, n):
        """JPEG byte tensors for frames start..start+n-1; indices outside the sequence are clamped (edge pad)."""
        idx = np.clip(np.arange(start, start + n), 0, len(self) - 1)
        cache = {}
        out = []
        for i in idx:
            i = int(i)
            if i not in cache:
                cache[i] = self.raw(i)
            out.append(cache[i])
        return out

    def close(self):
        mm = getattr(self.data, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass


def decode_frames(bufs, device):
    """List of JPEG byte tensors -> uint8 (n, 3, H, W) on `device` via CPU libjpeg-turbo + single DMA transfer."""
    return torch.stack([decode_jpeg(b) for b in bufs]).to(device, non_blocking=True)


def decode_batch(bufs_per_clip, device):
    """[[T byte tensors] x B] -> uint8 (B, T, 3, H, W) on device."""
    B, T = len(bufs_per_clip), len(bufs_per_clip[0])
    flat = [b for clip in bufs_per_clip for b in clip]
    x = decode_frames(flat, device)
    return x.view(B, T, *x.shape[1:])
