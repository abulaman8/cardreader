"""Iterative hard-negative mining on TRAIN games only.

Round r scans a rotating block of mine.games_per_round train games: caches each half temporarily, scores it,
keeps peaks with raw score >= mine.min_score that lie >= neg_exclusion_s from every card, deletes the temp
half, then caches the mined clips (frames/train/mined/).

  python -m src.v2.mine --ckpt runs/v2_r0/best.pt --round 1
  python -m src.v2.train --name v2_r1 --init runs/v2_r0/best.pt --mined data/v2/mined_round1.json
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from .common import get_device, keys_for_games, load_index, load_split
from .frames import extract_clips, extract_halves
from .infer import detections_from_scores, score_half
from .model import load_model_v2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--round", type=int, required=True, help="1, 2, ... (selects the rotating game block)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model_v2(args.ckpt, device)
    index, split = load_index(cfg), load_split(cfg)
    d, mc = cfg["data"], cfg["mine"]
    train = sorted(split["train"])
    assert not (set(train) & (set(split["val"]) | set(split["test"])))
    G = mc["games_per_round"]
    s0 = ((args.round - 1) * G) % len(train)
    games = (train + train)[s0:s0 + G]
    tmp_root = os.path.join(cfg["paths"]["frames"], "mine_tmp")
    half_win = d["cache_len"] / d["fps"] / 2
    excl = d["neg_exclusion_s"]
    mined = []
    for gi, key in enumerate(keys_for_games(games)):
        prefix = os.path.join(tmp_root, f"h{gi:04d}")
        extract_halves(cfg, index, [key], lambda k: prefix, args.workers)
        conf, disp = score_half(model, prefix, cfg, device)
        h = index["halves"][key]
        cards = np.array([c["t"] for c in h["cards"]], dtype=np.float64)
        keep = []
        for t, s in detections_from_scores(conf, disp, cfg):
            if s < mc["min_score"] or t < half_win or t > h["duration"] - half_win:
                continue
            if cards.size and np.min(np.abs(cards - t)) < excl:
                continue
            keep.append((t, s))
        keep = sorted(keep, key=lambda x: -x[1])[:mc["max_per_half"]]
        for t, s in keep:
            mined.append({"id": f"mined_r{args.round}_{len(mined):05d}", "key": key, "t": round(t, 2),
                          "start": round(t - half_win, 2), "score": round(s, 4), "kind": "mined"})
        for ext in (".data.npy", ".off.npy"):
            if os.path.exists(prefix + ext):
                os.remove(prefix + ext)
        import gc; gc.collect()
        print(f"[{gi + 1}/{2 * len(games)}] {key}: {len(keep)} hard negatives (total {len(mined)})")
    shutil.rmtree(tmp_root, ignore_errors=True)
    out = os.path.join(os.path.dirname(cfg["paths"]["index"]), f"mined_round{args.round}.json")
    with open(out, "w") as f:
        json.dump(mined, f)
    print(f"wrote {out} ({len(mined)} specs); caching clips ...")
    extract_clips(cfg, index, mined, "mined", args.workers)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
