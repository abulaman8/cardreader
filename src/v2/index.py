"""Build data/v2/index.json from the frozen benchmark split (400 train / 50 val / 50 test).

Training clips are mined ONLY from the 400 train games. Val clip specs are for train-time clip metrics.
Usage: python -m src.v2.index
"""
import argparse
import json
import os

import numpy as np

from ..index import build_halves, sample_negatives
from .common import add_v2_config_args, keys_for_games, load_config, load_split


def main():
    args = add_v2_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    d, root = cfg["data"], cfg["paths"]["data_root"]
    fps, L = d["fps"], d["cache_len"]
    ext = max(3 * cfg["targets"]["sigma"], cfg["targets"]["disp_radius"])   # card offsets kept around the window
    cache_s, half_win = L / fps, L / fps / 2
    rng = np.random.default_rng(d["seed"])

    split = load_split(cfg)
    train, val, test = split["train"], split["val"], split["test"]
    assert not (set(train) & (set(val) | set(test))), "train overlaps benchmark games"
    print(f"games: train {len(train)} | val {len(val)} | test {len(test)}; probing durations ...")
    halves, aux = build_halves(root, train + val + test)
    tkeys, vkeys = keys_for_games(train), keys_for_games(val)

    train_pos = []
    for key in tkeys:
        dur, vis = halves[key]["duration"], aux[key]["visible"]
        for t in vis:
            start = round(float(np.clip(t - half_win, 0.0, max(dur - cache_s, 0.0))), 2)
            rel = [int(round((c - start) * fps)) for c in vis if -ext <= (c - start) * fps < L + ext]
            train_pos.append({"id": f"pos_{len(train_pos):05d}", "key": key, "start": start,
                              "anchor": int(round((t - start) * fps)), "cards": rel})

    negs = sample_negatives(rng, tkeys, halves, aux, int(d["neg_pool"]), d["neg_mix"], d["neg_exclusion_s"], half_win)
    train_neg = [{"id": f"neg_{i:05d}", "key": n["key"], "t": n["t"], "kind": n["kind"],
                  "start": round(n["t"] - half_win, 2)} for i, n in enumerate(negs)]

    vpos = [{"key": k, "t": float(t), "u": float(rng.random())} for k in vkeys for t in aux[k]["visible"]]
    vneg = sample_negatives(rng, vkeys, halves, aux, int(round(d["eval_neg_per_pos"] * len(vpos))), d["neg_mix"],
                            d["neg_exclusion_s"], d["window"] / fps / 2)

    index = {"version": 2, "fps": fps, "cache_len": L, "splits": {"train": train, "val": val, "test": test},
             "halves": halves, "train_pos": train_pos, "train_neg": train_neg,
             "eval": {"val": {"pos": vpos, "neg": vneg}}}
    os.makedirs(os.path.dirname(cfg["paths"]["index"]) or ".", exist_ok=True)
    with open(cfg["paths"]["index"], "w") as f:
        json.dump(index, f)
    kinds = {}
    for n in train_neg:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    print(f"train positives {len(train_pos)} | negative pool {len(train_neg)} {kinds}")
    print(f"val clips: pos {len(vpos)} | neg {len(vneg)}")
    print(f"wrote {cfg['paths']['index']}")


if __name__ == "__main__":
    main()
