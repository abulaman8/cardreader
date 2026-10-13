"""Config loading with dotted CLI overrides, e.g. ``--set data.window=40 data.margin=12``."""
import copy

import yaml

DEFAULT_CONFIG = "configs/default.yaml"


def add_config_args(parser):
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config keys, e.g. --set data.window=40 train.lr=1e-4")
    return parser


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be KEY=VALUE, got {ov!r}")
        key, val = ov.split("=", 1)
        node, parts = cfg, key.split(".")
        for k in parts[:-1]:
            node = node[k]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def validate(cfg):
    d = cfg["data"]
    T, m = d["window"], d["margin"]
    if T > d["cache_len"]:
        raise ValueError(f"data.window ({T}) must be <= data.cache_len ({d['cache_len']})")
    if not (0 <= m and 2 * m < T):
        raise ValueError(f"data.margin ({m}) must satisfy 0 <= margin and 2*margin < window ({T})")
    return cfg


def load_config(path=DEFAULT_CONFIG, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return validate(apply_overrides(cfg, overrides))
