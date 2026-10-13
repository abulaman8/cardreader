"""FROZEN card-spotting benchmark. Do not edit: any change alters this file's sha256 and makes results
non-comparable with earlier rows in results.csv.

Model-agnostic: reads only SoccerNet-format prediction files and Labels-v2.json ground truth.
Imports nothing from the project code.

Prediction layout: <pred_dir>/<game>/results_spotting.json
  {"predictions": [{"gameTime": "1 - 12:34", "position": "<ms>", "half": "1", "label": "Card", "confidence": "0.93"}, ...]}

Usage:
  venv/bin/python benchmark/benchmark.py --run v2_r0 --val-pred runs/v2_r0/bench/calibrated/val \
      --test-pred runs/v2_r0/bench/calibrated/test

Rules:
- Ground truth: cards with visibility == "visible" (Yellow, Red, Yellow->red merged).
- "Not shown" cards are ignored: an unmatched prediction within tolerance of one is neither TP nor FP.
- Matching: greedy by descending confidence; a prediction is TP if an unmatched GT card in the same half is
  within |dt| <= tolerance seconds. Each GT card matches at most one prediction.
- Primary tolerance: +-5 s. +-1 s and +-2 s reported for information.
- Fixed thresholds: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95.
- Headline: test F1/P/R at +-5 s at the threshold maximizing val F1 at +-5 s (ties -> higher threshold),
  plus threshold-free test AP at +-5 s and loose a-mAP (delta 5..60 s, match |dt| <= delta/2).
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TOLERANCES = [1.0, 2.0, 5.0]
PRIMARY_TOL = 5.0
LOOSE_DELTAS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
CARD_LABELS = {"Yellow card", "Red card", "Yellow->red card"}
PRED_LABELS = CARD_LABELS | {"Card"}

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATH = os.path.join(HERE, "split.json")
RESULTS_CSV = os.path.join(HERE, "results.csv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_gt(data_root, game):
    with open(os.path.join(data_root, game, "Labels-v2.json")) as f:
        ann = json.load(f)["annotations"]
    vis, ign = {1: [], 2: []}, {1: [], 2: []}
    for a in ann:
        if a["label"] not in CARD_LABELS:
            continue
        half = int(a["gameTime"].split(" - ")[0].strip())
        t = int(a["position"]) / 1000.0
        (vis if a.get("visibility", "visible") == "visible" else ign)[half].append(t)
    return vis, ign


def load_preds(pred_dir, game):
    path = os.path.join(pred_dir, game, "results_spotting.json")
    if not os.path.exists(path):
        print(f"[warn] missing predictions: {path} (counted as no detections)")
        return []
    with open(path) as f:
        preds = json.load(f).get("predictions", [])
    out = []
    for p in preds:
        if p.get("label", "Card") not in PRED_LABELS:
            continue
        half = int(p["half"]) if "half" in p else int(p["gameTime"].split(" - ")[0].strip())
        out.append((half, int(float(p["position"])) / 1000.0, float(p["confidence"])))
    return out


def collect(data_root, pred_dir, games):
    dets, gts, ign = [], {}, {}
    for g in games:
        vis, nsh = load_gt(data_root, g)
        for h in (1, 2):
            gts[(g, h)] = np.asarray(vis[h], dtype=np.float64)
            ign[(g, h)] = np.asarray(nsh[h], dtype=np.float64)
        dets += [((g, h), t, s) for h, t, s in load_preds(pred_dir, g)]
    return dets, gts, ign


def match(dets, gts, ign, tol):
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    scores, tps = [], []
    for i in order:
        key, t, s = dets[i]
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
            scores.append(s)
            tps.append(1)
            continue
        ig = ign.get(key)
        if ig is not None and len(ig) and np.min(np.abs(ig - t)) <= tol:
            continue
        scores.append(s)
        tps.append(0)
    return np.asarray(scores), np.asarray(tps, dtype=np.int64), int(sum(len(v) for v in gts.values()))


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


def evaluate(dets, gts, ign):
    res = {"n_gt": int(sum(len(v) for v in gts.values())), "n_pred": len(dets), "grid": []}
    for tol in TOLERANCES:
        scores, tps, n_gt = match(dets, gts, ign, tol)
        if tol == PRIMARY_TOL:
            res["ap5"] = average_precision(tps, n_gt)
        for thr in THRESHOLDS:
            keep = scores >= thr - 1e-9
            tp = int(tps[keep].sum())
            fp = int(keep.sum()) - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / n_gt if n_gt else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            res["grid"].append({"thr": thr, "tol": tol, "precision": p, "recall": r, "f1": f1,
                                "tp": tp, "fp": fp, "fn": n_gt - tp})
    res["loose_mAP"] = float(np.mean([average_precision(match(dets, gts, ign, d / 2)[1], res["n_gt"])
                                      for d in LOOSE_DELTAS]))
    return res


def row(res, thr, tol):
    return next(r for r in res["grid"] if r["thr"] == thr and r["tol"] == tol)


def print_table(name, res):
    print(f"\n[{name}] GT cards {res['n_gt']} | predictions {res['n_pred']} | "
          f"AP@+-5s {res['ap5']:.4f} | loose a-mAP {res['loose_mAP']:.4f}")
    print("  thr  | " + " | ".join(f"+-{t:g}s  P     R     F1  " for t in TOLERANCES))
    for thr in THRESHOLDS:
        cells = [row(res, thr, t) for t in TOLERANCES]
        print(f"  {thr:.2f} | " + " | ".join(f"{c['precision']:.3f} {c['recall']:.3f} {c['f1']:.3f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name recorded in results.csv")
    p.add_argument("--val-pred", required=True)
    p.add_argument("--test-pred", required=True)
    p.add_argument("--data-root", default="soccernet_data")
    p.add_argument("--out", default=None, help="json report path (default: <test-pred>/../benchmark_<run>.json)")
    p.add_argument("--no-csv", action="store_true", help="do not append to results.csv")
    args = p.parse_args()
    if not os.path.exists(SPLIT_PATH):
        sys.exit("benchmark/split.json missing; run benchmark/make_split.py once")
    script_sha, split_sha = sha256(os.path.abspath(__file__)), sha256(SPLIT_PATH)
    print(f"benchmark.py sha256 {script_sha}\nsplit.json   sha256 {split_sha}")
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    val = evaluate(*collect(args.data_root, args.val_pred, split["val"]))
    test = evaluate(*collect(args.data_root, args.test_pred, split["test"]))
    print_table("val", val)
    print_table("test", test)

    val_rows = [r for r in val["grid"] if r["tol"] == PRIMARY_TOL]
    sel = max(val_rows, key=lambda r: (r["f1"], r["thr"]))["thr"]
    head = row(test, sel, PRIMARY_TOL)
    print(f"\nHEADLINE ({args.run}): threshold {sel:.2f} (val best F1 @ +-5s {row(val, sel, PRIMARY_TOL)['f1']:.4f})")
    print(f"  test @ +-5s: F1 {head['f1']:.4f} | P {head['precision']:.4f} | R {head['recall']:.4f} "
          f"(tp {head['tp']} fp {head['fp']} fn {head['fn']})")
    print(f"  test AP@+-5s {test['ap5']:.4f} | test loose a-mAP {test['loose_mAP']:.4f}")

    report = {"run": args.run, "time": datetime.datetime.now().isoformat(timespec="seconds"),
              "benchmark_sha256": script_sha, "split_sha256": split_sha, "val_pred": args.val_pred,
              "test_pred": args.test_pred, "selected_thr": sel, "headline": head, "val": val, "test": test}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.test_pred.rstrip("/"))),
                                   f"benchmark_{args.run}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"report: {out}")

    if not args.no_csv:
        new = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "run", "benchmark_sha256", "split_sha256", "selected_thr", "test_f1_5s",
                            "test_p_5s", "test_r_5s", "test_ap_5s", "test_loose_mAP", "val_f1_5s", "val_ap_5s",
                            "val_pred", "test_pred"])
            w.writerow([report["time"], args.run, script_sha, split_sha, sel, f"{head['f1']:.4f}",
                        f"{head['precision']:.4f}", f"{head['recall']:.4f}", f"{test['ap5']:.4f}",
                        f"{test['loose_mAP']:.4f}", f"{row(val, sel, PRIMARY_TOL)['f1']:.4f}", f"{val['ap5']:.4f}",
                        args.val_pred, args.test_pred])
        print(f"appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
