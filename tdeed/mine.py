"""Hard-negative mining for the T-DEED replica. Scans a rotating block of TRAIN games only.

  python -m tdeed.mine --ckpt runs/tdeed_r0/best.pt --round 1
  python -m tdeed.extract windows --kind mined --specs data/tdeed/mined_round1.json
  python -m tdeed.train --name tdeed_r1 --init runs/tdeed_r0/best.pt --mined data/tdeed/mined_round1.json

Per half: decode to a temp cache -> score -> NMS peaks with score >= mine.min_score whose window_s window
is entirely >= neg_exclusion_s from any card -> top mine.max_per_half -> window specs. Temp cache deleted.
Resumable: finished halves are appended to <out>.partial.jsonl and skipped on restart.
"""
import argparse
import json
import os

import numpy as np

from .common import get_device, keys_for_games, load_index
from .extract import cache_half
from .index import window_far_from_cards
from .infer import detections, score_half
from .model import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--out", default=None, help="default: data/tdeed/mined_round<r>.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device)
    index = load_index(cfg)
    d, mc = cfg["data"], cfg["mine"]
    fps, W, excl = d["fps"], float(d["window_s"]), d["neg_exclusion_s"]
    n_win = int(round(W * fps))

    train = index["splits"]["train"]
    G = min(int(mc["games_per_round"]), len(train))
    s0 = ((args.round - 1) * G) % len(train)
    games = (train + train)[s0:s0 + G]
    keys = keys_for_games(games)
    out = args.out or os.path.join(os.path.dirname(cfg["paths"]["index"]), f"mined_round{args.round}.json")
    partial = out + ".partial.jsonl"
    done = {}
    if os.path.exists(partial):
        with open(partial) as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r["windows"]
    print(f"round {args.round}: {len(games)} train games ({len(keys)} halves), {len(done)} halves already done")

    os.makedirs(mc["tmp_dir"], exist_ok=True)
    tmp = os.path.join(mc["tmp_dir"], "mine_half")
    for i, key in enumerate(keys):
        if key in done:
            continue
        hv = index["halves"][key]
        for ext in (".data.npy", ".off.npy"):
            if os.path.exists(tmp + ext):
                os.remove(tmp + ext)
        wins = []
        if cache_half(hv["video"], tmp, fps, d["frame_h"], d["frame_w"], d["jpeg_quality"]):
            prob = score_half(model, tmp, cfg, device)
            for t, s in detections(prob, cfg, min_score=mc["min_score"]):
                start = float(np.clip(t - W / 2, 0.0, max(hv["duration"] - W, 0.0)))
                if window_far_from_cards(hv, start, W, excl):
                    wins.append({"key": key, "t": round(t, 2), "start": round(start, 2), "score": round(s, 4)})
            wins = sorted(wins, key=lambda w: -w["score"])[:int(mc["max_per_half"])]
        with open(partial, "a") as f:
            f.write(json.dumps({"key": key, "windows": wins}) + "\n")
        done[key] = wins
        print(f"[{i + 1}/{len(keys)}] {key}: {len(wins)} hard negatives")
    for ext in (".data.npy", ".off.npy"):
        if os.path.exists(tmp + ext):
            os.remove(tmp + ext)

    specs = []
    for key in keys:
        for w in done.get(key, []):
            specs.append({"id": f"mined_r{args.round}_{len(specs):05d}", "key": key, "start": w["start"], "n": n_win,
                          "t": w["t"], "score": w["score"], "kind": "mined", "anchor": None, "cards": [], "ignore": []})
    tmp_out = out + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(specs, f)
    os.replace(tmp_out, out)
    print(f"wrote {len(specs)} mined window specs -> {out}")
    print(f"next: python -m tdeed.extract windows --kind mined --specs {out}")


if __name__ == "__main__":
    main()
