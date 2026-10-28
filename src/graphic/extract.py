"""Post-Event Broadcast Overlay Strip Extractor & Graphic Feature Cache.

Extracts post-event TV broadcast graphic strips across [t + 2.0s, t + 27.5s]
(K = 18 steps spaced by 1.5s) plus a pre-event baseline reference strip at
t - 3.0s:
  - Top scorebug / dropdown region: rows [6:42]   (36 px high)
  - Lower-third banner region:      rows [156:216] (60 px high)
  - Stacked strip: (3, 96, 398) resized to (3, 96, 256) uint8
  - Graphic auxiliary features g_aux: (18, 8) float16 capturing stationary
    pop-up yellow/red card badge intensity, area compactness, green substitution
    arrow suppression, and top/bottom banner appearance masks.

Usage:
  venv/bin/python -m src.graphic.extract
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import av
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.v2.common import half_prefix, load_config, load_index
from src.v2.frames import JpegSeq

K_STEPS = 18
STEP_OFFSETS_SEC = [2.0 + 1.5 * k for k in range(K_STEPS)]  # [2.0, 3.5, ..., 27.5]
PAIR_DELTA_SEC = 0.4
REF_OFFSET_SEC = -3.0
STRIP_W = 256
STRIP_H = 96


def slice_overlay_strip(frame_chw):
    """Extract (..., 3, 96, 398) overlay strip from (..., 3, 224, 398) RGB frame."""
    top = frame_chw[..., 6:42, :]
    bot = frame_chw[..., 156:216, :]
    return torch.cat([top, bot], dim=-2)


def compute_strip_and_gaux(f_ref_u8, f1_u8, f2_u8):
    """Given pre-event frame f_ref_u8 (3, 224, 398) and K post-event frame pairs
    f1_u8, f2_u8 of shape (K, 3, 224, 398) uint8, return:
      strips_u8: (K, 3, 96, 256) uint8
      ref_u8:    (1, 3, 96, 256) uint8
      g_aux:     (K, 8) float16
    """
    s_ref_raw = slice_overlay_strip(f_ref_u8.unsqueeze(0))  # (1, 3, 96, 398)
    s1_raw = slice_overlay_strip(f1_u8)                     # (K, 3, 96, 398)
    s2_raw = slice_overlay_strip(f2_u8)                     # (K, 3, 96, 398)

    s_ref = s_ref_raw.float() / 255.0
    s1 = s1_raw.float() / 255.0
    s2 = s2_raw.float() / 255.0

    # 1. Stationary 2D overlay mask (changes < 0.025 over 0.4s)
    stat = ((s1 - s2).abs().mean(dim=1) < 0.025).float()    # (K, 96, 398)
    # 2. Pop-up mask relative to pre-event reference t - 3s (excludes permanent team badges)
    popup = ((s1 - s_ref).abs().mean(dim=1) > 0.075).float()  # (K, 96, 398)
    mask = stat * popup

    r, g, b = s1[:, 0], s1[:, 1], s1[:, 2]
    # Saturated yellow card icon saliency
    yel = (
        torch.clamp(torch.minimum(r, g) - 1.35 * b, min=0.0)
        * ((r > 0.65) & (g > 0.55) & (b < 0.35)).float()
        * mask
    )
    # Saturated red card icon saliency
    red = (
        torch.clamp(r - 1.45 * torch.maximum(g, b), min=0.0)
        * ((r > 0.65) & (g < 0.30) & (b < 0.30)).float()
        * mask
    )
    # Saturated green substitution arrow/badge saliency
    grn = (
        torch.clamp(g - 1.25 * torch.maximum(r, b), min=0.0)
        * ((g > 0.50) & (r < 0.40)).float()
        * mask
    )

    y_pool = F.avg_pool2d(yel.unsqueeze(1), kernel_size=(8, 6), stride=(4, 3)).squeeze(1)
    r_pool = F.avg_pool2d(red.unsqueeze(1), kernel_size=(8, 6), stride=(4, 3)).squeeze(1)
    g_pool = F.avg_pool2d(grn.unsqueeze(1), kernel_size=(8, 6), stride=(4, 3)).squeeze(1)

    g0 = y_pool.amax(dim=(1, 2))                       # Peak stationary pop-up yellow badge
    g1 = (yel > 0.15).float().mean(dim=(1, 2))         # Yellow pop-up area fraction
    g2 = r_pool.amax(dim=(1, 2))                       # Peak stationary pop-up red badge
    g3 = (red > 0.15).float().mean(dim=(1, 2))         # Red pop-up area fraction
    g4 = g_pool.amax(dim=(1, 2))                       # Peak stationary pop-up green (substitution)
    g5 = mask[:, :36, :].mean(dim=(1, 2))              # Top scorebug pop-up fraction
    g6 = mask[:, 36:, :].mean(dim=(1, 2))              # Lower-third banner pop-up fraction
    g7 = (s1 - s_ref).abs().mean(dim=(1, 2, 3))        # Global strip diff vs pre-event

    g_aux = torch.stack([g0, g1, g2, g3, g4, g5, g6, g7], dim=-1).to(torch.float16)

    strips_u8 = (
        F.interpolate(s1_raw.float(), size=(STRIP_H, STRIP_W), mode="bilinear", align_corners=False)
        .clamp_(0, 255)
        .round_()
        .to(torch.uint8)
    )
    ref_u8 = (
        F.interpolate(s_ref_raw.float(), size=(STRIP_H, STRIP_W), mode="bilinear", align_corners=False)
        .clamp_(0, 255)
        .round_()
        .to(torch.uint8)
    )
    return strips_u8, ref_u8, g_aux


def _decode_mkv_half_group(task):
    """Worker function: opens one (game, half) MKV file and decodes all requested events."""
    game, half, events = task
    mkv_path = os.path.join("soccernet_data", game, f"{half}_224p.mkv")
    if not os.path.exists(mkv_path):
        return []

    try:
        container = av.open(mkv_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        dur = float(stream.duration * tb) if stream.duration else 3600.0
    except Exception:
        return []

    def grab_frame_at(target_sec):
        t_sec = max(0.0, min(target_sec, dur - 0.5))
        seek_ts = int(max(0.0, t_sec - 1.0) / tb)
        try:
            container.seek(seek_ts, backward=True, any_frame=False, stream=stream)
            best_arr = None
            for frame in container.decode(stream):
                pts_sec = float(frame.pts * tb) if frame.pts is not None else t_sec
                arr = frame.to_ndarray(format="rgb24")
                best_arr = arr
                if pts_sec >= t_sec - (0.5 / fps):
                    break
            if best_arr is None:
                return torch.zeros((3, 224, 398), dtype=torch.uint8)
            t_u8 = torch.from_numpy(best_arr).permute(2, 0, 1).contiguous()
            if t_u8.shape[1] != 224 or t_u8.shape[2] != 398:
                t_u8 = (
                    F.interpolate(t_u8.unsqueeze(0).float(), size=(224, 398), mode="bilinear", align_corners=False)
                    .squeeze(0)
                    .clamp_(0, 255)
                    .round_()
                    .to(torch.uint8)
                )
            return t_u8
        except Exception:
            return torch.zeros((3, 224, 398), dtype=torch.uint8)

    out_records = []
    for ev in events:
        t0 = float(ev["t"])
        f_ref = grab_frame_at(max(0.0, t0 + REF_OFFSET_SEC))
        f1_list, f2_list = [], []
        for off in STEP_OFFSETS_SEC:
            tk = t0 + off
            # Seek once per step and read tk and tk + 0.4s sequentially from the stream!
            t_sec = max(0.0, min(tk, dur - 0.6))
            t2_sec = max(0.0, min(tk + PAIR_DELTA_SEC, dur - 0.1))
            seek_ts = int(max(0.0, t_sec - 1.0) / tb)
            arr1, arr2 = None, None
            try:
                container.seek(seek_ts, backward=True, any_frame=False, stream=stream)
                for frame in container.decode(stream):
                    pts_sec = float(frame.pts * tb) if frame.pts is not None else t_sec
                    if arr1 is None and pts_sec >= t_sec - (0.5 / fps):
                        arr1 = frame.to_ndarray(format="rgb24")
                    if arr1 is not None and pts_sec >= t2_sec - (0.5 / fps):
                        arr2 = frame.to_ndarray(format="rgb24")
                        break
            except Exception:
                pass

            if arr1 is None:
                u1 = torch.zeros((3, 224, 398), dtype=torch.uint8)
            else:
                u1 = torch.from_numpy(arr1).permute(2, 0, 1).contiguous()
            if arr2 is None:
                u2 = u1.clone()
            else:
                u2 = torch.from_numpy(arr2).permute(2, 0, 1).contiguous()

            if u1.shape[1:] != (224, 398):
                u1 = F.interpolate(u1[None].float(), size=(224, 398), mode="bilinear", align_corners=False)[0].to(torch.uint8)
            if u2.shape[1:] != (224, 398):
                u2 = F.interpolate(u2[None].float(), size=(224, 398), mode="bilinear", align_corners=False)[0].to(torch.uint8)

            f1_list.append(u1)
            f2_list.append(u2)

        f1_u8 = torch.stack(f1_list, dim=0)
        f2_u8 = torch.stack(f2_list, dim=0)
        strips_u8, ref_u8, g_aux = compute_strip_and_gaux(f_ref, f1_u8, f2_u8)

        out_records.append({
            "id": ev["id"],
            "kind": ev["kind"],
            "label": int(ev["label"]),
            "game": game,
            "half": half,
            "t": t0,
            "strips": strips_u8,
            "ref": ref_u8,
            "g_aux": g_aux,
        })

    container.close()
    return out_records


def build_train_events(index, rng_seed=42):
    """Collect all positive card events + hard-negative Substitution, Goal, Mined, and Foul events
    guaranteed to be at least 35s away from any GT card in the same half.
    """
    rng = np.random.default_rng(rng_seed)
    train_games = index["splits"]["train"]

    # Map (game, half) -> list of card timestamps (in seconds)
    card_times = {}
    sub_events = []
    goal_events = []

    for g in train_games:
        lp = os.path.join("soccernet_data", g, "Labels-v2.json")
        if not os.path.exists(lp):
            continue
        with open(lp) as f:
            ann = json.load(f).get("annotations", [])
        for a in ann:
            gt_str = a.get("gameTime", "")
            if " - " not in gt_str:
                continue
            half = int(gt_str.split(" - ")[0])
            t_sec = int(a.get("position", "0")) / 1000.0
            lbl = a.get("label", "")
            if lbl in ("Yellow card", "Red card", "Yellow->red card"):
                card_times.setdefault((g, half), []).append(t_sec)
            elif lbl == "Substitution":
                sub_events.append((g, half, t_sec))
            elif lbl == "Goal":
                goal_events.append((g, half, t_sec))

    # Also register all train_pos timestamps into card_times
    pos_events = []
    for c in index["train_pos"]:
        g, h_str = c["key"].split("|")
        half = int(h_str)
        t_sec = float(c["start"]) + float(c["anchor"]) / 5.0
        card_times.setdefault((g, half), []).append(t_sec)
        pos_events.append({
            "id": c["id"],
            "kind": "pos",
            "label": 1,
            "game": g,
            "half": half,
            "t": t_sec,
        })

    def is_far_from_cards(g, half, t_sec, min_dist=35.0):
        for ct in card_times.get((g, half), []):
            if abs(t_sec - ct) < min_dist:
                return False
        return True

    # Filter and sample Substitution hard negatives (up to 1100)
    clean_subs = [(g, h, t) for (g, h, t) in sub_events if is_far_from_cards(g, h, t)]
    if len(clean_subs) > 1100:
        idx_sel = rng.choice(len(clean_subs), 1100, replace=False)
        clean_subs = [clean_subs[i] for i in idx_sel]

    neg_events = []
    for i, (g, h, t) in enumerate(clean_subs):
        neg_events.append({
            "id": f"sub_{i:05d}",
            "kind": "sub",
            "label": 0,
            "game": g,
            "half": h,
            "t": float(t),
        })

    # Filter and sample Goal hard negatives (up to 350)
    clean_goals = [(g, h, t) for (g, h, t) in goal_events if is_far_from_cards(g, h, t)]
    if len(clean_goals) > 350:
        idx_sel = rng.choice(len(clean_goals), 350, replace=False)
        clean_goals = [clean_goals[i] for i in idx_sel]

    for i, (g, h, t) in enumerate(clean_goals):
        neg_events.append({
            "id": f"goal_{i:05d}",
            "kind": "goal",
            "label": 0,
            "game": g,
            "half": h,
            "t": float(t),
        })

    # Mined hard negatives from round 1 & round 2 (up to 1300)
    mined_raw = []
    for mp_path in ("data/v2/mined_round1.json", "data/v2/mined_round2.json"):
        if os.path.exists(mp_path):
            with open(mp_path) as f:
                mined_raw.extend(json.load(f))
    clean_mined = []
    for c in mined_raw:
        g, h_str = c["key"].split("|")
        half = int(h_str)
        t_sec = float(c["t"])
        if is_far_from_cards(g, half, t_sec):
            clean_mined.append((c["id"], g, half, t_sec))
    if len(clean_mined) > 1300:
        idx_sel = rng.choice(len(clean_mined), 1300, replace=False)
        clean_mined = [clean_mined[i] for i in idx_sel]

    for cid, g, h, t in clean_mined:
        neg_events.append({
            "id": cid,
            "kind": "mined",
            "label": 0,
            "game": g,
            "half": h,
            "t": float(t),
        })

    # Foul & stoppage negatives from index['train_neg'] (up to 650)
    foul_stop = []
    for c in index["train_neg"]:
        if c.get("kind") in ("foul", "stoppage"):
            g, h_str = c["key"].split("|")
            half = int(h_str)
            t_sec = float(c["t"])
            if is_far_from_cards(g, half, t_sec):
                foul_stop.append((c["id"], c["kind"], g, half, t_sec))
    if len(foul_stop) > 650:
        idx_sel = rng.choice(len(foul_stop), 650, replace=False)
        foul_stop = [foul_stop[i] for i in idx_sel]

    for cid, knd, g, h, t in foul_stop:
        neg_events.append({
            "id": cid,
            "kind": knd,
            "label": 0,
            "game": g,
            "half": h,
            "t": float(t),
        })

    return pos_events + neg_events


def extract_eval_candidates(cfg, eval_cands_path, out_path):
    """Extract post-event graphic strips & g_aux for all Val & Test candidates in eval_cands.pt
    directly from the cached 5fps eval half JpegSeqs.
    """
    eval_cands = torch.load(eval_cands_path, map_location="cpu", weights_only=False)
    fps = float(cfg["data"]["fps"])  # 5.0 fps
    out_data = {}

    for sp_name in ("val", "test"):
        cands = eval_cands[sp_name]
        half_cache = {}
        records = []
        for c in tqdm(cands, desc=f"Graphic strips ({sp_name})"):
            g, h = c["game"], c["half"]
            key = f"{g}|{h}"
            if key not in half_cache:
                pre = half_prefix(cfg, sp_name, key)
                half_cache[key] = JpegSeq.open_half(pre)
            seq = half_cache[key]
            n_frames = len(seq)
            t0 = float(c["t"])

            ref_idx = int(np.clip(round((t0 + REF_OFFSET_SEC) * fps), 0, n_frames - 1))
            f_ref = seq.decode(ref_idx, 1)[0]

            f1_list, f2_list = [], []
            for off in STEP_OFFSETS_SEC:
                idx1 = int(np.clip(round((t0 + off) * fps), 0, n_frames - 1))
                idx2 = int(np.clip(idx1 + 2, 0, n_frames - 1))  # +0.4s at 5fps = +2 frames
                f1_list.append(seq.decode(idx1, 1)[0])
                f2_list.append(seq.decode(idx2, 1)[0])

            f1_u8 = torch.stack(f1_list, dim=0)
            f2_u8 = torch.stack(f2_list, dim=0)
            strips_u8, ref_u8, g_aux = compute_strip_and_gaux(f_ref, f1_u8, f2_u8)

            records.append({
                "game": g,
                "half": h,
                "t": t0,
                "strips": strips_u8,
                "ref": ref_u8,
                "g_aux": g_aux,
            })
        out_data[sp_name] = records

    torch.save(out_data, out_path)
    print(f"Saved Val ({len(out_data['val'])}) & Test ({len(out_data['test'])}) graphic strips -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--eval-cands", default="data/roi/eval_cands.pt")
    p.add_argument("--out-dir", default="data/graphic")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    index = load_index(cfg)

    train_out = os.path.join(args.out_dir, "train_strips.pt")
    if not os.path.exists(train_out):
        t_start = time.time()
        all_events = build_train_events(index)
        by_half = {}
        for ev in all_events:
            by_half.setdefault((ev["game"], ev["half"]), []).append(ev)
        # Sort events chronologically within each half so seeking moves mostly forward
        tasks = [(g, h, sorted(evs, key=lambda x: x["t"])) for (g, h), evs in by_half.items()]
        print(
            f"Extracting post-event graphic strips for {len(all_events)} train events "
            f"across {len(tasks)} halves using {args.workers} workers..."
        )
        records = []
        with mp.Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(_decode_mkv_half_group, tasks), total=len(tasks), desc="Train MKV halves"):
                records.extend(res)
        torch.save(records, train_out)
        dt = time.time() - t_start
        n_pos = sum(1 for r in records if r["label"] == 1)
        n_neg = sum(1 for r in records if r["label"] == 0)
        print(f"Saved {len(records)} train graphic strips ({n_pos} pos, {n_neg} neg) in {dt:.1f}s -> {train_out}")
    else:
        print(f"Found existing {train_out}, skipping train extraction.")

    eval_out = os.path.join(args.out_dir, "eval_strips.pt")
    if not os.path.exists(eval_out):
        extract_eval_candidates(cfg, args.eval_cands, eval_out)
    else:
        print(f"Found existing {eval_out}, skipping eval extraction.")


if __name__ == "__main__":
    main()
