"""Sliding-window inference on held-out halves -> SoccerNet-style predictions json per game.

  python -m src.infer --ckpt runs/<name>/best.pt --split test [--temperature 1.0] [--save-scores]
Output: runs/<name>/predictions/<split>/<game>/results_spotting.json
"""
import argparse
import json
import os

import numpy as np

from .common import get_device, half_feat_path, load_index, split_key, split_keys
from .model import load_model
from .spotting import nms_peaks, score_half


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=None, help="override infer.min_score")
    p.add_argument("--out", default=None)
    p.add_argument("--save-scores", action="store_true", help="also save per-frame score curves (.npy)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device)
    index = load_index(cfg)
    d, inf = cfg["data"], cfg["infer"]
    min_score = inf["min_score"] if args.min_score is None else args.min_score
    out_root = args.out or os.path.join(os.path.dirname(args.ckpt), "predictions", args.split)

    preds = {}
    for key in split_keys(index, args.split):
        path = half_feat_path(cfg, args.split, key)
        if not os.path.exists(path):
            print(f"[warn] missing features: {path}")
            continue
        game, half = split_key(key)
        scores = score_half(model, np.load(path, mmap_mode="r"), d["window"], inf["stride"], d["margin"], device,
                            inf["batch_windows"], args.temperature)
        if args.save_scores:
            os.makedirs(os.path.join(out_root, game), exist_ok=True)
            np.save(os.path.join(out_root, game, f"scores_half{half}.npy"), scores.astype(np.float16))
        for t, s in nms_peaks(scores, d["fps"], inf["nms_window_s"], min_score):
            preds.setdefault(game, []).append({
                "gameTime": f"{half} - {int(t // 60):02d}:{int(t % 60):02d}", "label": "Card",
                "position": str(int(round(t * 1000))), "half": str(half), "confidence": str(round(s, 5))})
    for game in index["splits"][args.split]:
        os.makedirs(os.path.join(out_root, game), exist_ok=True)
        with open(os.path.join(out_root, game, "results_spotting.json"), "w") as f:
            json.dump({"UrlLocal": game, "predictions": preds.get(game, [])}, f, indent=1)
    print(f"wrote predictions for {len(index['splits'][args.split])} games to {out_root}")


if __name__ == "__main__":
    main()
