"""Model-agnostic late fusion + dual calibration on SoccerNet-format predictions.

Reads <bench>/raw/{val,test}/<game>/results_spotting.json from one or more runs, fuses them, fits two
calibrators on VAL only, and writes benchmark-ready prediction dirs. Test labels are never read.

  # ensemble of two runs
  venv/bin/python -m ensemble.fuse --name ens_r2 --bench runs/v2_r2/bench runs/tdeed_r2/bench
  # single run: no fusion, recalibration only (adds the cumulative variant)
  venv/bin/python -m ensemble.fuse --name tdeed_r2_cal --bench runs/tdeed_r2/bench

Output (default runs/<name>/bench/):
  fused/{val,test}/            fused score before final calibration
  calibrated_marg/{val,test}/  isotonic P(this detection is a true card)      (current meaning of 0.70)
  calibrated_cum/{val,test}/   isotonic precision of all detections >= score   (new meaning of 0.70)
  fuse.json                    inputs, Platt params, weights, val AP, both calibrators

Steps:
1. Per input run: Platt scaling (1-D logistic on logit(raw score)) fitted on val detections. Puts runs on a
   common probability scale and keeps their ranking. Isotonic steps are not used here: ties lose ranking.
2. Per half: greedy clustering. Take the highest unclustered detection, add the best unclustered detection
   of every other run within +-tol. Fused time = score-weighted mean. Fused score = sum_m w_m * p_m
   (missing run = 0, so a detection only one run sees is down-weighted).
3. Weights: simplex grid (step 0.1), chosen by val AP@+-tol (threshold-free). Then NMS (+-nms_s/2).
4. Calibrators fitted on val fused detections, labels from benchmark.benchmark.match (same rule as the
   frozen benchmark: greedy, |dt| <= tol, 'not shown' cards ignored).
   - marg: isotonic regression of TP flag on score.
   - cum:  isotonic regression of cumulative precision (precision of all val detections with score >= s)
           on score. Threshold 0.70 then selects the largest set whose val precision is ~>= 0.70.
"""
import argparse
import itertools
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from benchmark.benchmark import SPLIT_PATH, average_precision, load_gt, load_preds, match

EPS = 1e-4


# ---------------------------------------------------------------- io
def read_run(bench, split, games, kind="raw"):
    """-> [((game, half), t, score)] from <bench>/<kind>/<split>/<game>/results_spotting.json"""
    pred_dir = os.path.join(bench, kind, split)
    return [((g, h), t, s) for g in games for h, t, s in load_preds(pred_dir, g)]


def val_labels(data_root, games):
    gts, ign = {}, {}
    for g in games:
        vis, nsh = load_gt(data_root, g)
        for h in (1, 2):
            gts[(g, h)] = np.asarray(vis[h], dtype=np.float64)
            ign[(g, h)] = np.asarray(nsh[h], dtype=np.float64)
    return gts, ign


def write_preds(out_root, games, dets):
    by_game = {}
    for (g, half), t, s in dets:
        by_game.setdefault(g, []).append({
            "gameTime": f"{half} - {int(t // 60):02d}:{int(t % 60):02d}", "label": "Card",
            "position": str(int(round(t * 1000))), "half": str(half), "confidence": f"{s:.5f}"})
    for g in games:
        os.makedirs(os.path.join(out_root, g), exist_ok=True)
        with open(os.path.join(out_root, g, "results_spotting.json"), "w") as f:
            json.dump({"UrlLocal": g, "predictions": by_game.get(g, [])}, f, indent=1)


# ---------------------------------------------------------------- calibration
def labelled(dets, gts, ign, tol):
    """benchmark.match drops ignored detections; returns (scores desc, tp flags, n_gt)."""
    return match(dets, gts, ign, tol)


def fit_platt(dets, gts, ign, tol):
    s, y, _ = labelled(dets, gts, ign, tol)
    x = np.log(np.clip(s, EPS, 1 - EPS) / (1 - np.clip(s, EPS, 1 - EPS))).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"a": 1.0, "b": 0.0}
    lr = LogisticRegression(C=1e4).fit(x, y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(p, s):
    s = np.clip(np.asarray(s, dtype=np.float64), EPS, 1 - EPS)
    return 1.0 / (1.0 + np.exp(-(p["a"] * np.log(s / (1 - s)) + p["b"])))


def fit_marginal(dets, gts, ign, tol):
    s, y, _ = labelled(dets, gts, ign, tol)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(s, y)
    return {"kind": "marginal", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
            "tol": tol, "n": int(len(s))}


def fit_cumulative(dets, gts, ign, tol):
    s, y, _ = labelled(dets, gts, ign, tol)          # sorted by score desc
    cum_prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(s, cum_prec)
    return {"kind": "cumulative", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
            "tol": tol, "n": int(len(s))}


def apply_iso(cal, s):
    return np.interp(np.asarray(s, dtype=np.float64), cal["x"], cal["y"])


# ---------------------------------------------------------------- fusion
def fuse(runs, weights, tol, nms_s):
    """runs: list of [((g,h), t, p)] with p on a common scale. Returns fused [((g,h), t, score)]."""
    by_key = {}
    for m, dets in enumerate(runs):
        for key, t, p in dets:
            by_key.setdefault(key, []).append((p, t, m))
    out = []
    for key, items in by_key.items():
        items.sort(key=lambda x: -x[0])
        used = [False] * len(items)
        clusters = []
        for i, (p, t, m) in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            members = {m: (p, t)}
            for j, (pj, tj, mj) in enumerate(items):      # items sorted desc: first hit per run is its best
                if used[j] or mj in members or abs(tj - t) > tol:
                    continue
                used[j] = True
                members[mj] = (pj, tj)
            ps = np.array([v[0] for v in members.values()])
            ts = np.array([v[1] for v in members.values()])
            score = float(sum(weights[mm] * pp for mm, (pp, _) in members.items()))
            clusters.append((float((ps * ts).sum() / max(ps.sum(), 1e-12)), score))
        clusters.sort(key=lambda c: -c[1])
        keep = []
        for t, s in clusters:
            if all(abs(t - u) > nms_s / 2 for u, _ in keep):
                keep.append((t, s))
        out += [(key, t, s) for t, s in keep]
    return out


def weight_grid(n, step=0.1):
    if n == 1:
        return [(1.0,)]
    k = int(round(1 / step))
    return [tuple(c / k for c in combo) for combo in itertools.product(range(k + 1), repeat=n)
            if sum(combo) == k]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output run name (benchmark rows: <name>_marg, <name>_cum)")
    ap.add_argument("--bench", nargs="+", required=True, help="input bench dirs, each with raw/{val,test}/")
    ap.add_argument("--out", default=None, help="default: runs/<name>/bench")
    ap.add_argument("--data-root", default="soccernet_data")
    ap.add_argument("--tol", type=float, default=5.0, help="matching / clustering tolerance (s)")
    ap.add_argument("--nms-s", type=float, default=10.0, help="final NMS window (s), suppresses +-nms_s/2")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="fixed weights (skip val search)")
    args = ap.parse_args()

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    out = args.out or os.path.join("runs", args.name, "bench")
    gts, ign = val_labels(args.data_root, split["val"])                 # val only; test labels never read

    raw = {s: [read_run(b, s, split[s]) for b in args.bench] for s in ("val", "test")}
    platt = [fit_platt(raw["val"][m], gts, ign, args.tol) for m in range(len(args.bench))]
    prob = {s: [[(k, t, float(p)) for (k, t, _), p in zip(raw[s][m], apply_platt(platt[m], [x[2] for x in raw[s][m]]))]
                for m in range(len(args.bench))] for s in raw}

    if args.weights:
        assert len(args.weights) == len(args.bench), "one weight per --bench"
        tot = sum(args.weights)
        weights, search = tuple(w / tot for w in args.weights), []
    else:
        search = []
        for w in weight_grid(len(args.bench)):
            fv = fuse(prob["val"], w, args.tol, args.nms_s)
            _, tps, n_gt = labelled(fv, gts, ign, args.tol)
            search.append({"w": list(w), "val_ap": average_precision(tps, n_gt)})
        uni = 1.0 / len(args.bench)
        best = max(search, key=lambda r: (round(r["val_ap"], 6), -sum(abs(x - uni) for x in r["w"])))
        weights = tuple(best["w"])
    fused = {s: fuse(prob[s], weights, args.tol, args.nms_s) for s in prob}
    _, tps, n_gt = labelled(fused["val"], gts, ign, args.tol)
    val_ap = average_precision(tps, n_gt)

    cals = {"marg": fit_marginal(fused["val"], gts, ign, args.tol),
            "cum": fit_cumulative(fused["val"], gts, ign, args.tol)}
    for s, dets in fused.items():
        write_preds(os.path.join(out, "fused", s), split[s], dets)
        for tag, cal in cals.items():
            cs = apply_iso(cal, [x[2] for x in dets])
            write_preds(os.path.join(out, f"calibrated_{tag}", s), split[s],
                        [(k, t, float(c)) for (k, t, _), c in zip(dets, cs)])

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fuse.json"), "w") as f:
        json.dump({"name": args.name, "inputs": args.bench, "tol": args.tol, "nms_s": args.nms_s,
                   "platt": platt, "weights": list(weights), "weight_search": search, "val_ap_fused": val_ap,
                   "n_dets": {s: len(d) for s, d in fused.items()}, "calibrators": cals}, f, indent=1)

    print(f"inputs: {args.bench}")
    print(f"weights {list(weights)} | val AP@+-{args.tol:g}s (fused, uncalibrated) {val_ap:.4f}")
    print(f"wrote {out}/{{fused,calibrated_marg,calibrated_cum}}/{{val,test}} + fuse.json")
    print("\nnext:")
    for tag in ("marg", "cum"):
        print(f"  venv/bin/python benchmark/benchmark.py --run {args.name}_{tag} "
              f"--val-pred {out}/calibrated_{tag}/val --test-pred {out}/calibrated_{tag}/test")


if __name__ == "__main__":
    main()
