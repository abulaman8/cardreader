"""End-to-end custom video evaluator and clip extractor for soccer card spotting.
Processes continuous full-match broadcast MP4s at 720x576 / 25 fps with custom event JSONs,
runs ens_r3_best_cum (CardNetV2 + T-DEED with Platt scaling and cumulative calibration),
matches predictions against ground truth card intervals, and generates 20-second clips.
"""
import argparse
import glob
import json
import os
import subprocess
import time
import av
import numpy as np
import torch
from tqdm import tqdm

from src.spotting import nms_peaks
from src.v2.model import load_model_v2
from tdeed.model import load_model as load_model_tdeed, vote as tdeed_vote

EPS = 1e-4


def apply_platt(p, s):
    s = np.clip(np.asarray(s, dtype=np.float64), EPS, 1.0 - EPS)
    return 1.0 / (1.0 + np.exp(-(p["a"] * np.log(s / (1.0 - s)) + p["b"])))


def apply_iso(cal, s):
    return np.interp(np.asarray(s, dtype=np.float64), cal["x"], cal["y"])


def load_gt_cards(json_path):
    """Load card events from custom JSON format."""
    with open(json_path) as f:
        d = json.load(f)
    evs = d.get("events", [])
    cards = []
    for e in evs:
        p_act = (e.get("primary_action_en") or e.get("primary_action") or "").lower()
        s_act = (e.get("secondary_action_en") or e.get("secondary_action") or "").lower()
        if "card" in p_act or "karte" in p_act or "card" in s_act or "karte" in s_act:
            fin = e.get("frame_in")
            fout = e.get("frame_out")
            if fin is not None and fout is not None:
                action = e.get("primary_action_en") or e.get("primary_action")
                if "card" not in (action or "").lower() and "karte" not in (action or "").lower():
                    action = e.get("secondary_action_en") or e.get("secondary_action") or "Card"
                player = e.get("secondary_player") if ("card" in s_act or "karte" in s_act) and e.get("secondary_player") else (e.get("primary_player") or "Unknown Player")
                team = e.get("secondary_team") if ("card" in s_act or "karte" in s_act) and e.get("secondary_team") else (e.get("primary_team") or "")
                cards.append({
                    "object_id": e.get("object_id", ""),
                    "frame_in": fin,
                    "frame_out": fout,
                    "t_in": fin / 25.0,
                    "t_out": fout / 25.0,
                    "t_mid": (fin + fout) / 50.0,
                    "action": action,
                    "player": player,
                    "team": team,
                    "description": e.get("description") or ""
                })
    return d.get("video_title", os.path.basename(json_path)), cards


def decode_video_5fps(video_path, chunk_frames_5fps=3000):
    """Generator yielding chunks of 5 fps frames resized to (224, 398, 3) uint8."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    buf = []
    chunk_start_idx = 0
    stream_idx = 0
    
    for frame in container.decode(stream):
        if stream_idx % 5 == 0:
            arr = frame.reformat(width=398, height=224, format="rgb24").to_ndarray()
            buf.append(arr)
            if len(buf) >= chunk_frames_5fps:
                yield chunk_start_idx, np.stack(buf)
                chunk_start_idx += len(buf)
                buf = []
        stream_idx += 1
        
    if buf:
        yield chunk_start_idx, np.stack(buf)
    container.close()


@torch.no_grad()
def run_cardnet_v2_stream(model, cfg, chunks_gen, total_len, device):
    """Run CardNetV2 inference over 5 fps video chunks."""
    T, m = cfg["data"]["window"], cfg["data"]["margin"]
    stride = cfg["infer"]["stride"]  # 10 frames = 2.0s
    R = cfg["targets"]["disp_radius"]  # 10 frames
    fps = cfg["data"]["fps"]  # 5
    
    acc_c = torch.zeros(total_len + T, device=device)
    acc_d = torch.zeros(total_len + T, device=device)
    cnt = torch.zeros(total_len + T, device=device)
    
    w = torch.full((T,), 1e-3, device=device)
    w[m:T - m] = 1.0
    ar = torch.arange(T, device=device)
    amp = device.type == "cuda"
    model.eval()
    
    window_buf = []
    buf_start = 0
    
    pbar = tqdm(total=total_len, desc="CardNetV2 scoring")
    for chunk_start, chunk_arr in chunks_gen:
        if len(window_buf) == 0:
            window_buf = chunk_arr
            buf_start = chunk_start
        else:
            window_buf = np.concatenate([window_buf, chunk_arr], axis=0)
            
        L = len(window_buf)
        if L >= T:
            valid_starts = [s for s in range(0, L - T + 1, stride)]
            if valid_starts:
                for i in range(0, len(valid_starts), 8):
                    sub = valid_starts[i:i + 8]
                    wins = np.stack([window_buf[s:s + T] for s in sub])  # (b, T, H, W, 3)
                    wins_t = torch.from_numpy(wins).permute(0, 1, 4, 2, 3).contiguous().to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        conf, disp, _ = model(wins_t)
                    p, dd = torch.sigmoid(conf.float()), disp.float()
                    
                    for k, s in enumerate(sub):
                        gidx = (buf_start + s) + ar
                        acc_c.index_add_(0, gidx, p[k] * w)
                        acc_d.index_add_(0, gidx, dd[k] * w)
                        cnt.index_add_(0, gidx, w)
                
                tail_start = valid_starts[-1] + stride
                buf_start += tail_start
                window_buf = window_buf[tail_start:]
        pbar.update(len(chunk_arr))
    pbar.close()
    
    cnt = cnt.clamp_min(1e-12)
    conf = (acc_c / cnt)[:total_len].cpu().numpy()
    disp = (acc_d / cnt)[:total_len].cpu().numpy()
    
    N = len(conf)
    tgt = np.clip(np.round(np.arange(N) + np.clip(disp, -1, 1) * R).astype(np.int64), 0, N - 1)
    votes = np.bincount(tgt, weights=conf, minlength=N)
    k = np.arange(-3, 4)
    kern = np.exp(-k ** 2 / 2.0)
    votes = np.convolve(votes, kern / kern.sum(), mode="same")
    
    out = []
    for t, _ in nms_peaks(votes, fps, cfg["infer"]["nms_window_s"], 1e-6):
        i = int(round(t * fps))
        s = float(conf[max(0, i - R):i + R + 1].max())
        if s >= cfg["infer"]["min_score"]:
            out.append((t, s))
    return out


@torch.no_grad()
def run_tdeed_stream(model, cfg, chunks_gen, total_len, device):
    """Run T-DEED inference over 5 fps video chunks."""
    T = int(cfg["data"]["clip_len"])  # 100 frames = 20.0s
    stride = int(round(T * (1.0 - cfg["infer"]["overlap"])))  # 50 frames = 10.0s
    fps = cfg["data"]["fps"]  # 5
    
    acc = torch.zeros(total_len + T, device=device)
    cnt = torch.zeros(total_len + T, device=device)
    ar = torch.arange(T, device=device)
    amp = device.type == "cuda"
    model.eval()
    
    window_buf = []
    buf_start = 0
    
    pbar = tqdm(total=total_len, desc="T-DEED scoring")
    for chunk_start, chunk_arr in chunks_gen:
        if len(window_buf) == 0:
            window_buf = chunk_arr
            buf_start = chunk_start
        else:
            window_buf = np.concatenate([window_buf, chunk_arr], axis=0)
            
        L = len(window_buf)
        if L >= T:
            valid_starts = [s for s in range(0, L - T + 1, stride)]
            if valid_starts:
                for i in range(0, len(valid_starts), 4):
                    sub = valid_starts[i:i + 4]
                    wins = np.stack([window_buf[s:s + T] for s in sub])  # (b, T, H, W, 3)
                    wins_t = torch.from_numpy(wins).permute(0, 1, 4, 2, 3).contiguous().to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        logits, disp = model(wins_t)
                    pv = torch.softmax(logits.float(), dim=-1)[..., 1]
                    if cfg["infer"]["vote"]:
                        pv = tdeed_vote(pv, disp.float())
                        
                    for k, s in enumerate(sub):
                        gidx = (buf_start + s) + ar
                        acc.index_add_(0, gidx, pv[k])
                        cnt.index_add_(0, gidx, torch.ones_like(pv[k]))
                        
                tail_start = valid_starts[-1] + stride
                buf_start += tail_start
                window_buf = window_buf[tail_start:]
        pbar.update(len(chunk_arr))
    pbar.close()
    
    prob = (acc / cnt.clamp_min(1e-12))[:total_len].cpu().numpy()
    return [(t, s) for t, s in nms_peaks(prob, fps, cfg["infer"]["nms_window_s"], cfg["infer"]["min_score"])]


def fuse_streams(dets_v2, dets_tdeed, fuse_cfg):
    """Late fuse v2 and tdeed detections using Platt scaling and weights."""
    platt_v2, platt_tdeed = fuse_cfg["platt"][0], fuse_cfg["platt"][1]
    w_v2, w_tdeed = fuse_cfg["weights"][0], fuse_cfg["weights"][1]
    tol = float(fuse_cfg["tol"]) / 2.0  # 2.5s clustering
    nms_s = float(fuse_cfg["nms_s"])
    
    items = []
    for t, s in dets_v2:
        items.append((apply_platt(platt_v2, s), t, 0))
    for t, s in dets_tdeed:
        items.append((apply_platt(platt_tdeed, s), t, 1))
        
    items.sort(key=lambda x: -x[0])
    used = [False] * len(items)
    clusters = []
    for i, (p, t, m) in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        members = {m: (p, t)}
        for j, (pj, tj, mj) in enumerate(items):
            if used[j] or mj in members or abs(tj - t) > tol:
                continue
            used[j] = True
            members[mj] = (pj, tj)
            
        ps = np.array([v[0] for v in members.values()])
        ts = np.array([v[1] for v in members.values()])
        weights = [w_v2, w_tdeed]
        fused_score = float(sum(weights[mm] * pp for mm, (pp, _) in members.items()))
        fused_t = float((ps * ts).sum() / max(ps.sum(), 1e-12))
        clusters.append((fused_t, fused_score))
        
    clusters.sort(key=lambda c: -c[1])
    keep = []
    for t, s in clusters:
        if all(abs(t - u) > nms_s / 2.0 for u, _ in keep):
            keep.append((t, s))
            
    cal_cum = fuse_cfg["calibrators"]["cum"]
    out = []
    for t, s_raw in keep:
        s_cal = float(apply_iso(cal_cum, s_raw))
        out.append((t, s_cal, s_raw))
    return out


def extract_clip(video_path, start_s, duration_s, out_path):
    """Extract a clean 20-second MP4 clip using ffmpeg."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ss = max(0.0, float(start_s))
    cmd = [
        "ffmpeg", "-y", "-ss", f"{ss:.2f}", "-i", video_path,
        "-t", f"{duration_s:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def process_single_game(video_path, json_path, args, fuse_cfg, device):
    """Process a single broadcast game end-to-end."""
    title, gt_cards = load_gt_cards(json_path)
    print(f"\n=======================================================")
    print(f"MATCH: {title}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Ground Truth Cards: {len(gt_cards)}")
    for idx, c in enumerate(gt_cards, 1):
        m, s = int(c["t_in"] // 60), int(c["t_in"] % 60)
        print(f"  [{idx}] {m:02d}:{s:02d} ({c['t_in']:.1f}s - {c['t_out']:.1f}s) | {c['action']} -> {c['player']} ({c['team']})")

    cont = av.open(video_path)
    st = cont.streams.video[0]
    duration_s = float(st.duration * st.time_base) if st.duration else float(st.frames or 217500) / 25.0
    total_5fps = int(round(duration_s * 5.0))
    cont.close()
    print(f"Duration: {duration_s/60:.1f} min ({total_5fps} frames @ 5 fps)")

    match_stem = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    cache_path = None
    if getattr(args, "cache_dir", None):
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(args.cache_dir, f"{match_stem}_peaks.pt")

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached peaks from {cache_path}...")
        cdata = torch.load(cache_path, map_location="cpu", weights_only=False)
        dets_v2 = cdata["dets_v2"]
        dets_tdeed = cdata["dets_tdeed"]
        print(f"Loaded: CardNetV2={len(dets_v2)} peaks, T-DEED={len(dets_tdeed)} peaks.")
    else:
        # 1. CardNetV2 Inference
        print("Loading CardNetV2...")
        v2_model, v2_cfg, _ = load_model_v2(args.ckpt_v2, device)
        t0 = time.time()
        dets_v2 = run_cardnet_v2_stream(v2_model, v2_cfg, decode_video_5fps(video_path), total_5fps, device)
        t1 = time.time()
        print(f"CardNetV2 completed in {t1 - t0:.1f}s: {len(dets_v2)} raw peaks.")
        del v2_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. T-DEED Inference
        print("Loading T-DEED...")
        tdeed_model, tdeed_cfg, _ = load_model_tdeed(args.ckpt_tdeed, device)
        t0 = time.time()
        dets_tdeed = run_tdeed_stream(tdeed_model, tdeed_cfg, decode_video_5fps(video_path), total_5fps, device)
        t1 = time.time()
        print(f"T-DEED completed in {t1 - t0:.1f}s: {len(dets_tdeed)} raw peaks.")
        del tdeed_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cache_path:
            torch.save({"dets_v2": dets_v2, "dets_tdeed": dets_tdeed}, cache_path)
            print(f"Saved candidate peaks -> {cache_path}")

    # 3. Fusion, Stage-2 Verification & Calibration
    s2_path = os.path.join(args.cache_dir, f"{match_stem}_stage2.pt") if getattr(args, "cache_dir", None) else None
    if not getattr(args, "stage1_only", False) and s2_path and os.path.exists(s2_path):
        from src.roi.rerank import apply_banner_ownership, build_feature_matrix
        with open(args.s2_cfg) as f:
            s2_cfg = json.load(f)
        s2 = torch.load(s2_path, map_location="cpu", weights_only=False)
        cands = s2["cands"]
        roi_raw = np.asarray(s2["roi_raw"], dtype=np.float64)
        v2c_raw = np.asarray(s2["v2c_raw"], dtype=np.float64)
        roi_p = apply_platt(s2_cfg["platt_roi"], roi_raw)
        v2c_p = apply_platt(s2_cfg["platt_v2c"], v2c_raw)
        p_v2 = np.asarray([c["p_v2"] for c in cands], dtype=np.float64)
        p_v2_eff = 0.80 * p_v2 + 0.20 * v2c_p
        p_td = np.asarray([c["p_td"] for c in cands], dtype=np.float64)
        p_td50 = np.asarray([c["p_td_50"] for c in cands], dtype=np.float64)
        res_roi = np.sqrt(np.maximum(p_v2_eff * roi_p, 0.0))
        s_pre = 0.56 * p_v2_eff + 0.22 * p_td + 0.03 * p_td50 + 0.08 * roi_p + 0.11 * res_roi
        g_owned = apply_banner_ownership(cands, s_pre, s2["g_out"], s2["g_steps"])
        graf_p = apply_platt(s2_cfg["platt_graf"], g_owned)
        X = build_feature_matrix(cands, roi_p, v2c_p, graf_p, b_c=0.20)
        w_s2 = np.asarray(s2_cfg["weights"], dtype=np.float64)
        raw_s2 = X @ w_s2
        cal_s2 = apply_iso(s2_cfg["calibrators"]["cum"], raw_s2)

        s_v2_raw = np.asarray([c["s_v2_raw"] for c in cands], dtype=np.float64)
        s_td_raw = np.asarray([c["s_td_raw"] for c in cands], dtype=np.float64)
        s1_cal = np.asarray([c["s_cal_s1"] for c in cands], dtype=np.float64)
        bug_stat = np.asarray([c.get("bug_stat", 0.5) for c in cands], dtype=np.float64)
        g_out = np.asarray(s2["g_out"], dtype=np.float64)
        g_yel = np.asarray([float(c["g_aux"][:, 0].max()) for c in cands], dtype=np.float64)
        g_red = np.asarray([float(c["g_aux"][:, 2].max()) for c in cands], dtype=np.float64)

        v2_strong = (s_v2_raw >= 0.895) & (v2c_raw >= 0.870) & (s1_cal >= 0.75)
        s2_confirm = (
            (g_out >= 0.40)
            | ((roi_raw >= 0.45) & (s_td_raw >= 0.25))
            | ((g_red >= 0.70) & ((s_td_raw >= 0.25) | (roi_raw >= 0.20)))
            | ((v2c_raw >= 0.935) & (g_yel >= 0.60))
        )
        final_cal = np.where(v2_strong & s2_confirm, np.maximum(cal_s2, 0.915), cal_s2)
        final_cal = np.where(bug_stat >= 0.15, final_cal, 0.0)

        cand_preds = sorted(
            [(float(c["t"]), float(sc), float(r)) for c, sc, r in zip(cands, final_cal, raw_s2) if sc >= args.threshold],
            key=lambda x: -x[1],
        )
        predictions = []
        for t_p, s_c, s_r in cand_preds:
            if all(abs(t_p - ut) > 6.0 for ut, _, _ in predictions):
                predictions.append((t_p, s_c, s_r))
        print(f"Stage-2 verified peaks >= {args.threshold:.2f}: {len(predictions)}")
    else:
        fused_dets = fuse_streams(dets_v2, dets_tdeed, fuse_cfg)
        predictions = [d for d in fused_dets if d[1] >= args.threshold]
        predictions.sort(key=lambda x: -x[1])
        print(f"Fused peaks >= {args.threshold:.2f}: {len(predictions)}")

    # 4. Greedy Matching
    matched_gt = set()
    tp_list, fp_list = [], []
    for pred_idx, (t_pred, s_cal, s_raw) in enumerate(predictions, 1):
        hit_gt_idx = None
        for gt_idx, c in enumerate(gt_cards):
            if gt_idx in matched_gt:
                continue
            if (c["t_in"] - args.tol) <= t_pred <= (c["t_out"] + args.tol):
                hit_gt_idx = gt_idx
                break
        if hit_gt_idx is not None:
            matched_gt.add(hit_gt_idx)
            tp_list.append({"pred_id": pred_idx, "t_pred": t_pred, "s_cal": s_cal, "s_raw": s_raw, "gt": gt_cards[hit_gt_idx]})
        else:
            fp_list.append({"pred_id": pred_idx, "t_pred": t_pred, "s_cal": s_cal, "s_raw": s_raw})

    fn_list = [c for idx, c in enumerate(gt_cards) if idx not in matched_gt]

    # Metrics
    prec = len(tp_list) / (len(tp_list) + len(fp_list)) if (len(tp_list) + len(fp_list)) else (1.0 if not len(gt_cards) else 0.0)
    rec = len(tp_list) / len(gt_cards) if len(gt_cards) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Game Summary: TP={len(tp_list)} FP={len(fp_list)} FN={len(fn_list)} | P={prec*100:.1f}% R={rec*100:.1f}% F1={f1:.4f}")

    # 5. Extract Clips
    match_stem = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    half_clip = args.clip_len / 2.0
    print("Slicing 20s clips...")

    for i, item in enumerate(tp_list, 1):
        t, c = item["t_pred"], item["gt"]
        m, s = int(t // 60), int(t % 60)
        plr = c["player"].replace(" ", "_").replace("/", "-")
        out_name = f"{match_stem}_TP_{i:02d}_{m:02d}m{s:02d}s_conf{item['s_cal']:.2f}_{plr}.mp4"
        out_path = os.path.join(args.out_dir, "tp", out_name)
        if getattr(args, "force_clips", False) or not os.path.exists(out_path):
            extract_clip(video_path, t - half_clip, args.clip_len, out_path)

    for i, item in enumerate(fp_list, 1):
        t = item["t_pred"]
        m, s = int(t // 60), int(t % 60)
        out_name = f"{match_stem}_FP_{i:02d}_{m:02d}m{s:02d}s_conf{item['s_cal']:.2f}.mp4"
        out_path = os.path.join(args.out_dir, "fp", out_name)
        if getattr(args, "force_clips", False) or not os.path.exists(out_path):
            extract_clip(video_path, t - half_clip, args.clip_len, out_path)

    for i, c in enumerate(fn_list, 1):
        t = c["t_mid"]
        m, s = int(t // 60), int(t % 60)
        plr = c["player"].replace(" ", "_").replace("/", "-")
        out_name = f"{match_stem}_FN_{i:02d}_{m:02d}m{s:02d}s_{plr}.mp4"
        out_path = os.path.join(args.out_dir, "fn", out_name)
        if getattr(args, "force_clips", False) or not os.path.exists(out_path):
            extract_clip(video_path, t - half_clip, args.clip_len, out_path)

    return {
        "title": title,
        "video": os.path.basename(video_path),
        "gt_count": len(gt_cards),
        "tp": len(tp_list),
        "fp": len(fp_list),
        "fn": len(fn_list),
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def find_well_annotated_games(custom_dir):
    """Find pairs of (mp4, json) for all 11 well-annotated games."""
    jsons = sorted(glob.glob(os.path.join(custom_dir, "*.json")))
    mp4s = sorted(glob.glob(os.path.join(custom_dir, "*.mp4")))
    valid = []
    for jf in jsons:
        with open(jf) as f:
            d = json.load(f)
        evs = d.get("events", [])
        if not evs:
            continue
        vtitle = d.get("video_title", "").strip()
        matches = [m for m in mp4s if vtitle and vtitle in os.path.basename(m)]
        if matches:
            valid.append((matches[0], jf))
    return valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default=None, help="Path to single broadcast MP4")
    p.add_argument("--json", default=None, help="Path to single custom annotation JSON")
    p.add_argument("--batch", default=None, help="Directory containing custom games to evaluate in batch")
    p.add_argument("--ckpt-v2", default="runs/v2_r2/best.pt")
    p.add_argument("--ckpt-tdeed", default="runs/tdeed_r3/best.pt")
    p.add_argument("--fuse-cfg", default="runs/ens_r3_best/bench/fuse.json")
    p.add_argument("--s2-cfg", default="runs/ens_r3_roi_graf/bench/fuse_roi_graf.json")
    p.add_argument("--stage1-only", action="store_true", help="Disable Stage-2 RoI+Graphic verification")
    p.add_argument("--out-dir", default="custom_data/clips")
    p.add_argument("--cache-dir", default="custom_data/cache_peaks", help="Directory to cache/load raw peaks per video")
    p.add_argument("--force-clips", action="store_true", help="Force re-extraction of clips even if existing")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--tol", type=float, default=5.0)
    p.add_argument("--clip-len", type=float, default=20.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.fuse_cfg) as f:
        fuse_cfg = json.load(f)

    if args.batch:
        games = find_well_annotated_games(args.batch)
        print(f"=== BATCH EVALUATION OVER {len(games)} WELL-ANNOTATED GAMES ===")
        print(f"Threshold: {args.threshold} | Tolerance: ±{args.tol}s | Clips -> {args.out_dir}\n")
        
        results = []
        for idx, (v_path, j_path) in enumerate(games, 1):
            print(f"\n[{idx}/{len(games)}] STARTING MATCH...")
            res = process_single_game(v_path, j_path, args, fuse_cfg, device)
            results.append(res)
            
        # Final Summary
        total_gt = sum(r["gt_count"] for r in results)
        total_tp = sum(r["tp"] for r in results)
        total_fp = sum(r["fp"] for r in results)
        total_fn = sum(r["fn"] for r in results)
        tot_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        tot_rec = total_tp / total_gt if total_gt else 0.0
        tot_f1 = 2 * tot_prec * tot_rec / (tot_prec + tot_rec) if (tot_prec + tot_rec) else 0.0

        print("\n" + "=" * 80)
        print(f"OVERALL BATCH EVALUATION RESULTS ({len(games)} GAMES):")
        print("=" * 80)
        for r in results:
            t = r["title"][:40]
            print(f"  {t:40s} | GT: {r['gt_count']:2d} | TP: {r['tp']:2d} | FP: {r['fp']:2d} | FN: {r['fn']:2d} | F1: {r['f1']:.3f}")
        print("-" * 80)
        print(f"TOTALS: GT={total_gt} | TP={total_tp} | FP={total_fp} | FN={total_fn}")
        print(f"FINAL METRICS: Precision = {tot_prec*100:.1f}% | Recall = {tot_rec*100:.1f}% | F1 = {tot_f1:.4f}")
        print("=" * 80)

        # Save results to json
        out_report = os.path.join(args.batch, "batch_evaluation_results.json")
        with open(out_report, "w") as f:
            json.dump({
                "threshold": args.threshold,
                "tolerance": args.tol,
                "n_games": len(games),
                "total_gt": total_gt,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "precision": tot_prec,
                "recall": tot_rec,
                "f1": tot_f1,
                "matches": results
            }, f, indent=2)
        print(f"Saved full batch report -> {out_report}")

    elif args.video and args.json:
        process_single_game(args.video, args.json, args, fuse_cfg, device)
    else:
        p.error("Must specify either --batch <dir> or both --video <mp4> and --json <json>")


if __name__ == "__main__":
    main()
