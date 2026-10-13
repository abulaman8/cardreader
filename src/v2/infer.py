"""v2 inference: sliding windows over cached halves, displacement voting, NMS, isotonic peak calibration,
SoccerNet-format predictions for the frozen benchmark.

  python -m src.v2.infer --ckpt runs/v2_r0/best.pt
  -> runs/v2_r0/bench/{raw,calibrated}/{val,test}/<game>/results_spotting.json + calibrator.json
  then: venv/bin/python benchmark/benchmark.py --run v2_r0 --val-pred runs/v2_r0/bench/calibrated/val \
            --test-pred runs/v2_r0/bench/calibrated/test
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..spotting import average_precision, best_threshold, match_detections, nms_peaks, spotting_map, \
    threshold_grid, window_starts
from .common import get_device, gt_dicts, half_prefix, keys_for_games, load_index, split_key
from .frames import JpegSeq, half_len
from .model import load_model_v2


class WindowGroups(Dataset):
    """Groups of consecutive windows; frames for a group are decoded once in a worker."""

    def __init__(self, prefix, starts, T, group):
        self.prefix, self.T = prefix, T
        self.groups = [starts[i:i + group] for i in range(0, len(starts), group)]
        self._seq = None

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, g):
        if self._seq is None:
            self._seq = JpegSeq.open_half(self.prefix)
        sts = self.groups[g]
        f0 = sts[0]
        frames = self._seq.decode(f0, sts[-1] + self.T - f0)
        return frames, torch.tensor(sts) - f0, f0

    def close(self):
        if self._seq is not None:
            self._seq.close()
            self._seq = None


@torch.no_grad()
def score_half(model, prefix, cfg, device, stride=None):
    """Per-frame confidence (sigmoid) and displacement for a cached half. Core-weighted window averaging."""
    d, inf = cfg["data"], cfg["infer"]
    T, m = d["window"], d["margin"]
    stride = stride or inf["stride"]
    N = half_len(prefix)
    L = max(N, T)
    starts = window_starts(L, T, stride)
    w = torch.full((T,), 1e-3, device=device)
    w[m:T - m] = 1.0
    acc_c, acc_d, cnt = torch.zeros(L, device=device), torch.zeros(L, device=device), torch.zeros(L, device=device)
    ar = torch.arange(T, device=device)
    amp = device.type == "cuda"
    model.eval()
    ds = WindowGroups(prefix, starts, T, inf["group_windows"])
    loader = DataLoader(ds, batch_size=None,
                        num_workers=inf["num_workers"], prefetch_factor=2 if inf["num_workers"] else None)
    for frames, local, f0 in loader:
        local_dev = local.to(device)
        idx = local_dev[:, None] + ar[None, :]                              # (G, T) into frames
        frames_dev = frames.to(device, non_blocking=True)
        for i in range(0, idx.shape[0], inf["windows_per_batch"]):
            sub = idx[i:i + inf["windows_per_batch"]]
            win = frames_dev[sub]                                           # (b, T, 3, H, W) uint8
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                conf, disp, _ = model(win)
            p, dd = torch.sigmoid(conf.float()), disp.float()
            gidx = (sub + int(f0)).flatten()
            acc_c.index_add_(0, gidx, (p * w).flatten())
            acc_d.index_add_(0, gidx, (dd * w).flatten())
            cnt.index_add_(0, gidx, w.expand_as(p).flatten())
    ds.close()
    cnt = cnt.clamp_min(1e-12)
    return (acc_c / cnt)[:N].cpu().numpy(), (acc_d / cnt)[:N].cpu().numpy()


def detections_from_scores(conf, disp, cfg, vote=None):
    """Peaks [(t_seconds, score)]. With voting, frames vote for t + disp*R weighted by confidence; peaks are
    located on the smoothed vote map and scored by the max confidence within +-R (stays in [0, 1])."""
    fps, inf, R = cfg["data"]["fps"], cfg["infer"], cfg["targets"]["disp_radius"]
    vote = inf["vote"] if vote is None else vote
    if not vote:
        return nms_peaks(conf, fps, inf["nms_window_s"], inf["min_score"])
    N = len(conf)
    tgt = np.clip(np.round(np.arange(N) + np.clip(disp, -1, 1) * R).astype(np.int64), 0, N - 1)
    votes = np.bincount(tgt, weights=conf, minlength=N)
    k = np.arange(-3, 4)
    kern = np.exp(-k ** 2 / 2.0)
    votes = np.convolve(votes, kern / kern.sum(), mode="same")
    out = []
    for t, _ in nms_peaks(votes, fps, inf["nms_window_s"], 1e-6):
        i = int(round(t * fps))
        s = float(conf[max(0, i - R):i + R + 1].max())
        if s >= inf["min_score"]:
            out.append((t, s))
    return out


def run_spotting(model, cfg, split, keys, device, stride=None, progress=True):
    dets = []
    for key in tqdm(keys, desc=f"spot {split}", disable=not progress):
        pre = half_prefix(cfg, split, key)
        if not os.path.exists(pre + ".off.npy"):
            print(f"[warn] half not cached: {pre}")
            continue
        conf, disp = score_half(model, pre, cfg, device, stride)
        dets += [(key, t, s) for t, s in detections_from_scores(conf, disp, cfg)]
    return dets


def spot_metrics(dets, gts, ign, cfg):
    ev = cfg["eval"]
    tol = float(ev["tolerance_s"])
    _, tps, n_gt = match_detections(dets, gts, ign, tol)
    grid = threshold_grid(dets, gts, ign, ev["thresholds"], [tol])
    best = best_threshold(grid, tol)
    return {"ap5": average_precision(tps, n_gt), "loose_mAP": spotting_map(dets, gts, ign, ev["loose_deltas"])[0],
            "best_thr": best["thr"], "best_f1": best["f1"], "best_precision": best["precision"],
            "best_recall": best["recall"], "n_dets": len(dets), "n_gt": n_gt}


def fit_calibrator(dets, gts, ign, tol):
    """Isotonic map raw peak score -> P(detection is a true card within +-tol s). Fit on val only."""
    scores, tps, _ = match_detections(dets, gts, ign, tol)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(scores, tps)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(), "tol": tol, "n": int(len(scores))}


def apply_calibrator(cal, scores):
    return np.interp(np.asarray(scores, dtype=np.float64), cal["x"], cal["y"])


def write_predictions(out_root, games, dets):
    by_game = {}
    for key, t, s in dets:
        game, half = split_key(key)
        by_game.setdefault(game, []).append({
            "gameTime": f"{half} - {int(t // 60):02d}:{int(t % 60):02d}", "label": "Card",
            "position": str(int(round(t * 1000))), "half": str(half), "confidence": f"{s:.5f}"})
    for g in games:
        os.makedirs(os.path.join(out_root, g), exist_ok=True)
        with open(os.path.join(out_root, g, "results_spotting.json"), "w") as f:
            json.dump({"UrlLocal": g, "predictions": by_game.get(g, [])}, f, indent=1)


def write_bench(out, index_splits, raw, gts_val, ign_val, tol):
    """Fit calibrator on val dets, write raw + calibrated predictions for val and test."""
    cal = fit_calibrator(raw["val"], gts_val, ign_val, tol)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calibrator.json"), "w") as f:
        json.dump(cal, f)
    for split, dets in raw.items():
        write_predictions(os.path.join(out, "raw", split), index_splits[split], dets)
        cs = apply_calibrator(cal, [s for _, _, s in dets])
        write_predictions(os.path.join(out, "calibrated", split), index_splits[split],
                          [(k, t, float(c)) for (k, t, _), c in zip(dets, cs)])
    return cal


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="default: <ckpt dir>/bench")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model_v2(args.ckpt, device)
    index = load_index(cfg)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "bench")
    raw = {s: run_spotting(model, cfg, s, keys_for_games(index["splits"][s]), device, args.stride)
           for s in ("val", "test")}
    gts, ign = gt_dicts(index, keys_for_games(index["splits"]["val"]))
    print("val (raw scores):", spot_metrics(raw["val"], gts, ign, cfg))
    cal = write_bench(out, index["splits"], raw, gts, ign, float(cfg["eval"]["tolerance_s"]))
    print(f"calibrator fitted on {cal['n']} val detections -> {out}/calibrator.json")
    run = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    print(f"\nnext:\n  venv/bin/python benchmark/benchmark.py --run {run} "
          f"--val-pred {out}/calibrated/val --test-pred {out}/calibrated/test")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
