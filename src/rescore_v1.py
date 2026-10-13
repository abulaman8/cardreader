"""Re-score v1 (frozen DINOv2 + temporal transformer, 10 fps) on the frozen 50/50 benchmark.

1. Extracts v1 features for any benchmark val/test half not yet cached (the 20 v1 games already are).
2. Runs v1 sliding-window inference with v1's own settings.
3. Fits the same isotonic calibration on val, writes raw + calibrated predictions.

  python -m src.rescore_v1 --ckpt runs/v1/best.pt
  venv/bin/python benchmark/benchmark.py --run v1 --val-pred runs/v1/bench/calibrated/val \
      --test-pred runs/v1/bench/calibrated/test
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import get_device, half_feat_path
from .encoder import DinoEncoder
from .extract import extract_halves
from .index import build_halves
from .model import load_model
from .spotting import nms_peaks, score_half
from .v2.common import keys_for_games
from .v2.infer import write_bench

SPLIT = "benchmark/split.json"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/v1/best.pt")
    p.add_argument("--out", default=None, help="default: <ckpt dir>/bench")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device)
    with open(SPLIT) as f:
        split = json.load(f)
    games = {"val": split["val"], "test": split["test"]}
    halves, _ = build_halves(cfg["paths"]["data_root"], games["val"] + games["test"])
    pseudo_index = {"splits": games, "halves": halves}

    if any(not os.path.exists(half_feat_path(cfg, s, k)) for s in games for k in keys_for_games(games[s])):
        enc = DinoEncoder(cfg["encoder"]["name"], device, cfg["encoder"]["batch_frames"])
        for s in games:
            extract_halves(cfg, pseudo_index, enc, s)
        del enc
        torch.cuda.empty_cache()

    d, inf = cfg["data"], cfg["infer"]
    raw = {}
    for s in games:
        dets = []
        for key in keys_for_games(games[s]):
            feats = np.load(half_feat_path(cfg, s, key), mmap_mode="r")
            sc = score_half(model, feats, d["window"], inf["stride"], d["margin"], device, inf["batch_windows"])
            dets += [(key, t, v) for t, v in nms_peaks(sc, d["fps"], inf["nms_window_s"], inf["min_score"])]
        raw[s] = dets
    vkeys = keys_for_games(games["val"])
    gts = {k: np.array([c["t"] for c in halves[k]["cards"] if c["visibility"] == "visible"]) for k in vkeys}
    ign = {k: np.array([c["t"] for c in halves[k]["cards"] if c["visibility"] != "visible"]) for k in vkeys}
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "bench")
    cal = write_bench(out, games, raw, gts, ign, 5.0)
    print(f"calibrator fitted on {cal['n']} val detections; predictions in {out}/{{raw,calibrated}}/{{val,test}}")
    print(f"next:\n  venv/bin/python benchmark/benchmark.py --run v1 --val-pred {out}/calibrated/val "
          f"--test-pred {out}/calibrated/test")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
