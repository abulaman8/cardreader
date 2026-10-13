"""Evaluation on held-out games (20 val / 20 test).

Library: evaluate_split() is also called by train.py every epoch (val).
CLI (final): fit temperature on val clips, pick threshold on val (best F1 at select tolerance),
apply once to test. Reports raw and temperature-calibrated variants.

  python -m src.evaluate --ckpt runs/<name>/best.pt
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .common import card_times, get_device, half_feat_path, load_index, split_keys
from .dataset import EvalClips
from .model import load_model
from .spotting import best_threshold, nms_peaks, score_half, spotting_map, threshold_grid


def _safe(fn, y, s):
    try:
        return float(fn(y, s)) if len(np.unique(y)) == 2 else float("nan")
    except ValueError:
        return float("nan")


@torch.no_grad()
def clip_metrics(model, clips, cfg, device, batch_size=256):
    """Clip-level metrics on EvalClips. Returns (metrics, frame_logits, frame_targets)."""
    model.eval()
    cw = cfg["targets"]["clip_weight"]
    fls, cls, ys, cs = [], [], [], []
    amp = device.type == "cuda"
    for x, y, c in DataLoader(clips, batch_size=batch_size, shuffle=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            fl, cl = model(x.to(device))
        fls.append(fl.float().cpu()); cls.append(cl.float().cpu()); ys.append(y); cs.append(c)
    fl, cl, y, c = torch.cat(fls), torch.cat(cls), torch.cat(ys), torch.cat(cs)
    loss_f = F.binary_cross_entropy_with_logits(fl, y).item()
    loss_c = F.binary_cross_entropy_with_logits(cl, c).item()
    cn, clip_p, fmax_p = c.numpy(), torch.sigmoid(cl).numpy(), torch.sigmoid(fl).max(1).values.numpy()
    m = {"loss": loss_f + cw * loss_c, "loss_frame": loss_f, "loss_clip": loss_c,
         "clip_auc": _safe(roc_auc_score, cn, clip_p), "clip_ap": _safe(average_precision_score, cn, clip_p),
         "fmax_auc": _safe(roc_auc_score, cn, fmax_p), "fmax_ap": _safe(average_precision_score, cn, fmax_p),
         "n_clips": int(len(cn)), "n_pos": int(cn.sum())}
    return m, fl, y


def fit_temperature(frame_logits, frame_targets):
    """Single scalar temperature minimising frame BCE on held-out clips."""
    z, y = frame_logits.flatten().float(), frame_targets.flatten().float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def evaluate_split(model, cfg, index, split, device, temperature=1.0, clips=None, return_dets=False):
    d, inf, ev = cfg["data"], cfg["infer"], cfg["eval"]
    fps = d["fps"]
    dets, gts, ign = [], {}, {}
    n_halves = n_missing = 0
    for key in split_keys(index, split):
        path = half_feat_path(cfg, split, key)
        if not os.path.exists(path):
            n_missing += 1
            continue
        scores = score_half(model, np.load(path, mmap_mode="r"), d["window"], inf["stride"], d["margin"], device,
                            inf["batch_windows"], temperature)
        dets += [(key, t, s) for t, s in nms_peaks(scores, fps, inf["nms_window_s"], inf["min_score"])]
        gts[key] = np.asarray(card_times(index, key, True), dtype=np.float64)
        ign[key] = np.asarray(card_times(index, key, False), dtype=np.float64)
        n_halves += 1
    res = {"n_halves": n_halves, "n_missing_halves": n_missing, "n_dets": len(dets),
           "n_gt": int(sum(len(v) for v in gts.values())), "temperature": temperature}
    if n_halves:
        res["tight_mAP"], res["tight_per_delta"] = spotting_map(dets, gts, ign, ev["tight_deltas"])
        res["loose_mAP"], res["loose_per_delta"] = spotting_map(dets, gts, ign, ev["loose_deltas"])
        res["grid"] = threshold_grid(dets, gts, ign, ev["thresholds"], ev["grid_tolerances_s"])
        res["best"] = best_threshold(res["grid"], ev["select_tolerance_s"])
    else:
        res["tight_mAP"] = res["loose_mAP"] = float("nan")
    if clips is None:
        clips = EvalClips(cfg, index, split)
    if len(clips):
        res["clip"] = clip_metrics(model, clips, cfg, device)[0]
    if return_dets:
        res["detections"] = [{"key": k, "t": t, "score": s} for k, t, s in dets]
    return res


def flatten_metrics(res, prefix):
    """Scalars for logs / tensorboard."""
    out = {}
    for k in ("tight_mAP", "loose_mAP", "n_dets", "n_gt", "n_halves"):
        if k in res:
            out[f"{prefix}/{k}"] = res[k]
    for k, v in res.get("clip", {}).items():
        out[f"{prefix}/{k}"] = v
    if res.get("best"):
        b = res["best"]
        out.update({f"{prefix}/best_thr": b["thr"], f"{prefix}/best_f1": b["f1"],
                    f"{prefix}/best_precision": b["precision"], f"{prefix}/best_recall": b["recall"]})
    for r in res.get("grid", []):
        tag = f"{prefix}/grid/thr{r['thr']:.2f}_tol{r['tol']:g}"
        out.update({f"{tag}/precision": r["precision"], f"{tag}/recall": r["recall"], f"{tag}/f1": r["f1"]})
    return out


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="output json (default: <ckpt>_eval.json)")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, ckpt = load_model(args.ckpt, device)
    index = load_index(cfg)
    sel_tol = cfg["eval"]["select_tolerance_s"]

    val_clips, test_clips = EvalClips(cfg, index, "val"), EvalClips(cfg, index, "test")
    _, val_fl, val_y = clip_metrics(model, val_clips, cfg, device)
    temp = fit_temperature(val_fl, val_y)
    print(f"fitted temperature on val clips: {temp:.4f}")

    report = {"ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "temperature": temp, "variants": {}}
    for name, t in (("raw", 1.0), ("calibrated", temp)):
        val = evaluate_split(model, cfg, index, "val", device, t, val_clips)
        test = evaluate_split(model, cfg, index, "test", device, t, test_clips)
        thr = val["best"]["thr"] if val.get("best") else None
        test_at_thr = [r for r in test.get("grid", []) if r["thr"] == thr]
        report["variants"][name] = {"selected_thr": thr, "val": val, "test": test, "test_at_selected_thr": test_at_thr}
        print(f"\n[{name}] T={t:.3f}  selected thr (val best F1 @ +-{sel_tol}s): {thr}")
        print(f"  val : tight {_fmt(val['tight_mAP'])} loose {_fmt(val['loose_mAP'])} "
              f"clip_auc {_fmt(val.get('clip', {}).get('clip_auc'))}")
        print(f"  test: tight {_fmt(test['tight_mAP'])} loose {_fmt(test['loose_mAP'])} "
              f"clip_auc {_fmt(test.get('clip', {}).get('clip_auc'))}")
        for r in test_at_thr:
            print(f"  test @thr {r['thr']:.2f} +-{r['tol']:g}s: P {r['precision']:.3f} R {r['recall']:.3f} "
                  f"F1 {r['f1']:.3f} (tp {r['tp']} fp {r['fp']} fn {r['fn']})")

    out = args.out or os.path.splitext(args.ckpt)[0] + "_eval.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
