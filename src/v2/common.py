"""v2 shared helpers: config, split/index access, frame-cache paths, ground truth."""
import json
import os

import numpy as np

from ..common import HALVES, card_times, get_device, half_key, split_key  # noqa: F401
from ..config import load_config  # noqa: F401

V2_CONFIG = "configs/v2.yaml"


def add_v2_config_args(parser):
    parser.add_argument("--config", default=V2_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser


def load_split(cfg):
    with open(cfg["paths"]["split"]) as f:
        return json.load(f)


def load_index(cfg):
    with open(cfg["paths"]["index"]) as f:
        return json.load(f)


def keys_for_games(games):
    return [half_key(g, h) for g in games for h in HALVES]


def half_prefix(cfg, split, key):
    """Frame-cache prefix of a full half: <prefix>.data.npy (JPEG bytes) + <prefix>.off.npy (offsets)."""
    game, half = split_key(key)
    return os.path.join(cfg["paths"]["frames"], "eval", split, game, f"half{half}")


def clip_path(cfg, kind, clip_id):
    return os.path.join(cfg["paths"]["frames"], "train", kind, f"{clip_id}.npz")


def gt_dicts(index, keys):
    gts = {k: np.asarray(card_times(index, k, True), dtype=np.float64) for k in keys}
    ign = {k: np.asarray(card_times(index, k, False), dtype=np.float64) for k in keys}
    return gts, ign
