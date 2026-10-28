"""Two-Stage Candidate Re-Ranker & Benchmark Emitter (Stage-1 Ens + Stage-2 RoI + Stage-2 GraphicNet).

Scores the top-K Stage-1 candidates per half using:
  1. RefereeRoIVerifier (128x128 referee tubelet around [-2.5s, +2.5s] with Flip TTA)
  2. BroadcastGraphicVerifier (top scorebug + lower-third overlay strips across
     [+2.0s, +27.5s] relative to t - 3.0s with Flip TTA and Banner Ownership gating)
Optimizes the Stage-1 + Stage-2 RoI + Stage-2 Graphic fusion weights strictly on VAL,
fits Marginal & Cumulative Isotonic Calibrators on VAL, writes SoccerNet-format
prediction directories, and runs benchmark/benchmark.py.

Usage:
  venv/bin/python -m src.roi.rerank --name ens_r3_roi_graf
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch

from benchmark.benchmark import SPLIT_PATH, average_precision, match
from ensemble.fuse import (
    apply_iso,
    apply_platt,
    fit_cumulative,
    fit_marginal,
    fit_platt,
    read_run,
    val_labels,
    write_preds,
)
from src.graphic.model import BroadcastGraphicVerifier
from src.roi.model import RefereeRoIVerifier
from src.v2.common import get_device


@torch.no_grad()
def score_candidates(model, cands, device, batch_size=64, tta_flip=True):
    model.eval()
    probs_clip, probs_peak = [], []
    for s in range(0, len(cands), batch_size):
        sub = cands[s:s + batch_size]
        crops = torch.stack([c["crop"] for c in sub]).to(device, non_blocking=True)
        aux = torch.stack([c["aux"].float() for c in sub]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            f_logits, c_logit = model(crops, aux)
            p_c = torch.sigmoid(c_logit.float())
            p_f = torch.sigmoid(f_logits.float())
            if tta_flip:
                f_logits_fl, c_logit_fl = model(crops.flip(-1), aux)
                p_c = 0.5 * (p_c + torch.sigmoid(c_logit_fl.float()))
                p_f = 0.5 * (p_f + torch.sigmoid(f_logits_fl.float()))

        p_c = p_c.cpu().numpy()
        p_f = p_f.cpu().numpy()
        for k, c in enumerate(sub):
            probs_clip.append(float(p_c[k]))
            off = c["center_offset"]
            r0, r1 = max(0, off - 4), min(p_f.shape[1], off + 5)
            probs_peak.append(float(p_f[k, r0:r1].max()))

    probs_clip = np.asarray(probs_clip, dtype=np.float64)
    probs_peak = np.asarray(probs_peak, dtype=np.float64)
    return 0.75 * probs_clip + 0.25 * probs_peak


@torch.no_grad()
def score_graphic_strips(model, strips_list, device, batch_size=64, tta_flip=True):
    model.eval()
    p_out_list, p_step_list = [], []
    for s in range(0, len(strips_list), batch_size):
        sub = strips_list[s:s + batch_size]
        strips = torch.stack([x["strips"] for x in sub]).to(device, non_blocking=True)
        ref = torch.stack([x["ref"] for x in sub]).to(device, non_blocking=True)
        g_aux = torch.stack([x["g_aux"].float() for x in sub]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            s_log, o_log = model(strips, ref, g_aux)
            p_o = torch.sigmoid(o_log.float())
            p_s = torch.sigmoid(s_log.float())
            if tta_flip:
                s_log_f, o_log_f = model(strips.flip(-1), ref.flip(-1), g_aux)
                p_o = 0.5 * (p_o + torch.sigmoid(o_log_f.float()))
                p_s = 0.5 * (p_s + torch.sigmoid(s_log_f.float()))
        p_out_list.append(p_o.cpu().numpy())
        p_step_list.append(p_s.cpu().numpy())
    return np.concatenate(p_out_list), np.concatenate(p_step_list, axis=0)


def extract_centered_v2_raw(cands):
    """Centered 25-frame full-frame V2 score extracted around candidate timestamp t (stored in aux[:, 0])."""
    out = []
    for c in cands:
        off = c["center_offset"]
        v2_f = c["aux"][:, 0].float().numpy()
        peak_c = float(v2_f[max(0, off - 4):min(len(v2_f), off + 5)].max())
        peak_all = float(v2_f.max())
        out.append(0.60 * peak_c + 0.40 * peak_all)
    return np.asarray(out, dtype=np.float64)


def apply_banner_ownership(cands, s_pre, g_out, g_steps):
    """Compute post-event broadcast graphic score with Banner Ownership gating:
    when two candidates in the same half fall within 27.5s of each other, a post-event
    broadcast card banner at t_banner belongs to the stronger preceding candidate.
    """
    g_burst2 = 0.5 * (g_steps[:, :-1] + g_steps[:, 1:])
    p_raw = 0.65 * g_out + 0.20 * g_burst2.max(axis=1) + 0.15 * g_steps.max(axis=1)

    by_gh = {}
    for i, c in enumerate(cands):
        by_gh.setdefault((c["game"], c["half"]), []).append(i)

    own_gate = np.ones(len(cands), dtype=np.float64)
    for _, idxs in by_gh.items():
        for i in idxs:
            ti = cands[i]["t"]
            si = s_pre[i]
            k_star = int(np.argmax(g_burst2[i]))
            t_banner = ti + 2.0 + 1.5 * (k_star + 0.5)
            for j in idxs:
                if i == j:
                    continue
                tj = cands[j]["t"]
                sj = s_pre[j]
                if sj > si + 0.015 and abs(tj - ti) <= 27.5:
                    if (tj <= t_banner - 1.0 and t_banner - tj <= 28.0) or abs(tj - ti) <= 16.0:
                        own_gate[i] = min(own_gate[i], 0.08)

    return p_raw * own_gate


def build_feature_matrix(cands, p_roi, p_v2c, p_graf, b_c=0.20):
    """Features for each candidate:
    f0: p_v2_eff ((1 - b_c)*p_v2 + b_c*p_v2c: Stage-1 sliding-window + Stage-2 centered-window V2)
    f1: p_td (Platt-scaled T-DEED within 2.5s)
    f2: p_td_50 (Platt-scaled T-DEED within 5.0s, soft context)
    f3: p_roi (Platt-scaled Stage-2 128x128 Referee RoI Verifier)
    f4: res_roi = sqrt(p_v2_eff * p_roi) (V2 + RoI joint confirmation when T-DEED is silent)
    f5: p_graf (Platt-scaled Stage-2 Broadcast Card Graphic Verifier with Banner Ownership)
    f6: res_graf = sqrt(max(p_v2_eff, p_td) * p_graf) (Stoppage/Gesture + Broadcast Graphic confirmation)
    f7: res_3way = (p_v2_eff * p_roi * p_graf)**(1/3) (Three-way agreement boost)
    """
    p_v2 = np.asarray([c["p_v2"] for c in cands], dtype=np.float64)
    p_v2_eff = (1.0 - b_c) * p_v2 + b_c * p_v2c
    p_td = np.asarray([c["p_td"] for c in cands], dtype=np.float64)
    p_td_50 = np.asarray([c["p_td_50"] for c in cands], dtype=np.float64)
    res_roi = np.sqrt(np.maximum(p_v2_eff * p_roi, 0.0))
    res_graf = np.sqrt(np.maximum(np.maximum(p_v2_eff, p_td) * p_graf, 0.0))
    res_3way = np.maximum(p_v2_eff * p_roi * p_graf, 0.0) ** (1.0 / 3.0)
    return np.stack([p_v2_eff, p_td, p_td_50, p_roi, res_roi, p_graf, res_graf, res_3way], axis=1)


def append_tail_detections(reranked_dets, split_name, games):
    """Append rank > 25 tail detections from ens_r3_best scaled below the re-ranked top-25 candidates."""
    ens_all = read_run("runs/ens_r3_best/bench", split_name, games, kind="fused")
    reranked_by_gh = {}
    for gh, t, s in reranked_dets:
        reranked_by_gh.setdefault(gh, []).append((t, s))

    min_top_score = min((s for _, _, s in reranked_dets), default=0.01)
    out = list(reranked_dets)
    for gh, t, s in ens_all:
        existing = reranked_by_gh.get(gh, [])
        if any(abs(t - et) <= 1e-3 for et, _ in existing):
            continue
        out.append((gh, float(t), float(min(s, min_top_score * 0.95) * 0.5)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/roi_v1/best.pt")
    p.add_argument("--ckpt-graphic", default="runs/graphic_v1/best.pt")
    p.add_argument("--eval-cands", default="data/roi/eval_cands.pt")
    p.add_argument("--eval-strips", default="data/graphic/eval_strips.pt")
    p.add_argument("--stage2-cache", default="data/graphic/stage2_scores.pt")
    p.add_argument("--name", default="ens_r3_roi_graf")
    p.add_argument("--tol", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = get_device(args.device)
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    val_games, test_games = split["val"], split["test"]
    gts_val, ign_val = val_labels("soccernet_data", val_games)

    eval_cands = torch.load(args.eval_cands, map_location="cpu", weights_only=False)
    val_cands = eval_cands["val"]
    test_cands = eval_cands["test"]

    if args.stage2_cache and os.path.exists(args.stage2_cache):
        print(f"Loading cached Stage-2 RoI & Graphic scores from {args.stage2_cache}...", flush=True)
        s2 = torch.load(args.stage2_cache, map_location="cpu", weights_only=False)
        roi_val_raw, v2c_val_raw = s2["val"]["roi_raw"], s2["val"]["v2c_raw"]
        g_val_out, g_val_steps = s2["val"]["g_out"], s2["val"]["g_steps"]
        roi_test_raw, v2c_test_raw = s2["test"]["roi_raw"], s2["test"]["v2c_raw"]
        g_test_out, g_test_steps = s2["test"]["g_out"], s2["test"]["g_steps"]
    else:
        print("Scoring Val & Test candidates with Stage-2 RoI & Graphic Verifiers (+ Flip TTA)...", flush=True)
        roi_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        roi_model = RefereeRoIVerifier(pretrained=False).to(device)
        roi_model.load_state_dict(roi_ckpt["model"])
        roi_val_raw = score_candidates(roi_model, val_cands, device, batch_size=64, tta_flip=True)
        roi_test_raw = score_candidates(roi_model, test_cands, device, batch_size=64, tta_flip=True)
        v2c_val_raw = extract_centered_v2_raw(val_cands)
        v2c_test_raw = extract_centered_v2_raw(test_cands)

        graf_ckpt = torch.load(args.ckpt_graphic, map_location="cpu", weights_only=False)
        graf_model = BroadcastGraphicVerifier(pretrained=False).to(device)
        graf_model.load_state_dict(graf_ckpt["model"])
        eval_strips = torch.load(args.eval_strips, map_location="cpu", weights_only=False)
        g_val_out, g_val_steps = score_graphic_strips(graf_model, eval_strips["val"], device)
        g_test_out, g_test_steps = score_graphic_strips(graf_model, eval_strips["test"], device)

    # Fit Platt scaling on Stage-2 RoI & Centered-V2 using Val candidates only
    val_roi_dets = [((c["game"], c["half"]), c["t"], float(s)) for c, s in zip(val_cands, roi_val_raw)]
    platt_roi = fit_platt(val_roi_dets, gts_val, ign_val, args.tol)
    roi_val_p = apply_platt(platt_roi, roi_val_raw)
    roi_test_p = apply_platt(platt_roi, roi_test_raw)

    val_v2c_dets = [((c["game"], c["half"]), c["t"], float(s)) for c, s in zip(val_cands, v2c_val_raw)]
    platt_v2c = fit_platt(val_v2c_dets, gts_val, ign_val, args.tol)
    v2c_val_p = apply_platt(platt_v2c, v2c_val_raw)
    v2c_test_p = apply_platt(platt_v2c, v2c_test_raw)

    # Pre-graphic Stage-1 + RoI score for Banner Ownership ordering
    def compute_s_pre(cands, roi_p, v2c_p):
        p_v2 = np.asarray([c["p_v2"] for c in cands], dtype=np.float64)
        p_v2_eff = 0.80 * p_v2 + 0.20 * v2c_p
        p_td = np.asarray([c["p_td"] for c in cands], dtype=np.float64)
        p_td50 = np.asarray([c["p_td_50"] for c in cands], dtype=np.float64)
        res_roi = np.sqrt(np.maximum(p_v2_eff * roi_p, 0.0))
        return 0.56 * p_v2_eff + 0.22 * p_td + 0.03 * p_td50 + 0.08 * roi_p + 0.11 * res_roi

    s_pre_val = compute_s_pre(val_cands, roi_val_p, v2c_val_p)
    s_pre_test = compute_s_pre(test_cands, roi_test_p, v2c_test_p)

    g_val_owned = apply_banner_ownership(val_cands, s_pre_val, g_val_out, g_val_steps)
    g_test_owned = apply_banner_ownership(test_cands, s_pre_test, g_test_out, g_test_steps)

    val_graf_dets = [((c["game"], c["half"]), c["t"], float(s)) for c, s in zip(val_cands, g_val_owned)]
    platt_graf = fit_platt(val_graf_dets, gts_val, ign_val, args.tol)
    graf_val_p = apply_platt(platt_graf, g_val_owned)
    graf_test_p = apply_platt(platt_graf, g_test_owned)

    X_val = build_feature_matrix(val_cands, roi_val_p, v2c_val_p, graf_val_p, b_c=0.20)
    X_test = build_feature_matrix(test_cands, roi_test_p, v2c_test_p, graf_test_p, b_c=0.20)

    # Grid search fusion weights strictly on VAL (composite of Val AP@+-5s and Val peak F1)
    best_obj = -1.0
    best_ap5 = -1.0
    best_f1 = -1.0
    best_w = None
    for w_roi in (0.03, 0.06, 0.09):
        for w_rroi in (0.04, 0.08, 0.12):
            for w_graf in (0.04, 0.08, 0.12):
                for w_rgraf in (0.08, 0.12, 0.16, 0.20):
                    for w_3way in (0.04, 0.08, 0.12):
                        w = np.array([0.55, 0.20, 0.02, w_roi, w_rroi, w_graf, w_rgraf, w_3way], dtype=np.float64)
                        w = w / w.sum()
                        scores = X_val @ w
                        dets = [
                            ((c["game"], c["half"]), c["t"], float(sc))
                            for c, sc in zip(val_cands, scores)
                        ]
                        _, tp_v, n_gt_v = match(dets, gts_val, ign_val, args.tol)
                        ap5 = average_precision(tp_v, n_gt_v)
                        cum_tp = np.cumsum(tp_v)
                        ks = np.arange(1, len(tp_v) + 1)
                        pr = cum_tp / ks
                        rc = cum_tp / n_gt_v
                        f1_curve = 2.0 * pr * rc / np.maximum(pr + rc, 1e-9)
                        max_f1 = float(f1_curve.max())
                        obj = 0.5 * ap5 + 0.5 * max_f1
                        if obj > best_obj:
                            best_obj = obj
                            best_ap5 = ap5
                            best_f1 = max_f1
                            best_w = w

    print(
        f"Best Val AP@+-5s (top-25): {best_ap5:.4f} | Val Peak F1: {best_f1:.4f} | "
        f"Weights [v2, td, td50, roi, res_roi, graf, res_graf, res_3way]: {np.round(best_w, 4).tolist()}",
        flush=True,
    )

    val_scores = X_val @ best_w
    test_scores = X_test @ best_w

    fused_val_top = [((c["game"], c["half"]), c["t"], float(sc)) for c, sc in zip(val_cands, val_scores)]
    fused_test_top = [((c["game"], c["half"]), c["t"], float(sc)) for c, sc in zip(test_cands, test_scores)]

    fused_val = append_tail_detections(fused_val_top, "val", val_games)
    fused_test = append_tail_detections(fused_test_top, "test", test_games)

    cal_marg = fit_marginal(fused_val, gts_val, ign_val, args.tol)
    cal_cum = fit_cumulative(fused_val, gts_val, ign_val, args.tol)

    out_bench = os.path.join("runs", args.name, "bench")
    for sp_name, games, dets in (("val", val_games, fused_val), ("test", test_games, fused_test)):
        write_preds(os.path.join(out_bench, "fused", sp_name), games, dets)
        write_preds(
            os.path.join(out_bench, "calibrated_marg", sp_name),
            games,
            [(gh, t, float(apply_iso(cal_marg, s))) for gh, t, s in dets],
        )
        write_preds(
            os.path.join(out_bench, "calibrated_cum", sp_name),
            games,
            [(gh, t, float(apply_iso(cal_cum, s))) for gh, t, s in dets],
        )

    with open(os.path.join(out_bench, "fuse_roi_graf.json"), "w") as f:
        json.dump(
            {
                "name": args.name,
                "ckpt_roi": args.ckpt,
                "ckpt_graphic": args.ckpt_graphic,
                "val_ap5": best_ap5,
                "val_peak_f1": best_f1,
                "weights": best_w.tolist(),
                "platt_roi": platt_roi,
                "platt_graf": platt_graf,
                "calibrators": {"marg": cal_marg, "cum": cal_cum},
            },
            f,
            indent=2,
        )

    for tag in ("marg", "cum"):
        cmd = [
            "./venv/bin/python",
            "-m",
            "benchmark.benchmark",
            "--run",
            f"{args.name}_{tag}",
            "--val-pred",
            os.path.join(out_bench, f"calibrated_{tag}", "val"),
            "--test-pred",
            os.path.join(out_bench, f"calibrated_{tag}", "test"),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
