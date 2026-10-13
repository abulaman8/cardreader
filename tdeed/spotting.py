"""Peak NMS, spotting metrics, isotonic calibration and SoccerNet-format writers (single class 'Card').

Metric definitions mirror benchmark/benchmark.py (match |dt| <= tol, greedy by score, 'not shown' cards
ignored). The frozen benchmark script remains the source of truth for reported numbers.
"""
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .common import split_key


def window_starts(N, T, stride):
    if N <= T:
        return [0]
    s = list(range(0, N - T + 1, stride))
    if s[-1] != N - T:
        s.append(N - T)
    return s


def nms_peaks(scores, fps, window_s, min_score):
    """Greedy: take max, suppress +-window_s/2, repeat. Returns [(t_seconds, score)]."""
    s = np.array(scores, dtype=np.float32, copy=True)
    r = int(round(window_s * fps / 2))
    out = []
    while s.size:
        i = int(s.argmax())
        v = float(s[i])
        if v < min_score:
            break
        out.append((i / fps, v))
        s[max(0, i - r):i + r + 1] = -1.0
    return out


def match_detections(dets, gts, ignores, tol):
    """dets [(key, t, score)], gts/ignores {key: times}. Returns (scores, tp_flags, n_gt) sorted by score desc."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    scores, tps = [], []
    for i in order:
        key, t, sc = dets[i]
        g = gts.get(key)
        hit = -1
        if g is not None and len(g):
            d = np.abs(g - t)
            d[used[key]] = np.inf
            j = int(d.argmin())
            if d[j] <= tol:
                hit = j
        if hit >= 0:
            used[key][hit] = True
            scores.append(sc)
            tps.append(1)
            continue
        ig = ignores.get(key)
        if ig is not None and len(ig) and np.min(np.abs(ig - t)) <= tol:
            continue
        scores.append(sc)
        tps.append(0)
    n_gt = int(sum(len(v) for v in gts.values()))
    return np.asarray(scores, dtype=np.float64), np.asarray(tps, dtype=np.int64), n_gt


def average_precision(tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(tps) == 0:
        return 0.0
    tp, fp = np.cumsum(tps), np.cumsum(1 - tps)
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def spotting_map(dets, gts, ignores, deltas):
    n_gt = int(sum(len(v) for v in gts.values()))
    per = {float(d): average_precision(match_detections(dets, gts, ignores, d / 2)[1], n_gt) for d in deltas}
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def threshold_grid(dets, gts, ignores, thresholds, tolerances):
    rows = []
    for tol in tolerances:
        scores, tps, n_gt = match_detections(dets, gts, ignores, tol)
        for thr in thresholds:
            keep = scores >= thr
            tp = int(tps[keep].sum())
            fp = int(keep.sum()) - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / n_gt if n_gt else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            rows.append({"thr": round(float(thr), 2), "tol": float(tol), "precision": p, "recall": r, "f1": f1,
                         "tp": tp, "fp": fp, "fn": n_gt - tp})
    return rows


def best_threshold(grid, tol):
    rows = [r for r in grid if r["tol"] == float(tol)]
    return max(rows, key=lambda r: (r["f1"], r["thr"])) if rows else None


def spot_metrics(dets, gts, ign, cfg):
    ev = cfg["eval"]
    tol = float(ev["tolerance_s"])
    _, tps, n_gt = match_detections(dets, gts, ign, tol)
    best = best_threshold(threshold_grid(dets, gts, ign, ev["thresholds"], [tol]), tol)
    return {"ap5": average_precision(tps, n_gt), "loose_mAP": spotting_map(dets, gts, ign, ev["loose_deltas"])[0],
            "best_thr": best["thr"], "best_f1": best["f1"], "best_precision": best["precision"],
            "best_recall": best["recall"], "n_dets": len(dets), "n_gt": n_gt}


# ---------------------------------------------------------------- calibration + writers
def fit_calibrator(dets, gts, ign, tol):
    """Isotonic map raw peak score -> P(true card within +-tol). Fit on val only."""
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


def write_bench(out, splits, raw, gts_val, ign_val, tol):
    """Fit calibrator on val detections; write raw + calibrated predictions for every split in `raw`."""
    cal = fit_calibrator(raw["val"], gts_val, ign_val, tol)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calibrator.json"), "w") as f:
        json.dump(cal, f)
    for split, dets in raw.items():
        write_predictions(os.path.join(out, "raw", split), splits[split], dets)
        cs = apply_calibrator(cal, [s for _, _, s in dets])
        write_predictions(os.path.join(out, "calibrated", split), splits[split],
                          [(k, t, float(c)) for (k, t, _), c in zip(dets, cs)])
    return cal
