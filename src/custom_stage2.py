"""Stage-2 (RefereeRoIVerifier + BroadcastGraphicVerifier) extractor and re-ranker for custom broadcast matches.

Operates on cached Stage-1 peaks in `custom_data/cache_peaks/<stem>_peaks.pt`,
extracts 25-frame 5fps clips (for RefereeRoIVerifier & Centered-V2) and 18-step
post-event broadcast overlay strips (for BroadcastGraphicVerifier) in a single
seek per candidate, scores all candidates on GPU, and caches Stage-2 features to
`custom_data/cache_peaks/<stem>_stage2.pt`.
"""
import argparse
import concurrent.futures as cf
import json
import os
import time

import av
import numpy as np
import torch

from src.custom_eval import (
    apply_iso,
    apply_platt,
    extract_clip,
    find_well_annotated_games,
    fuse_streams,
    load_gt_cards,
)
from src.graphic.extract import (
    PAIR_DELTA_SEC,
    REF_OFFSET_SEC,
    STEP_OFFSETS_SEC,
    compute_strip_and_gaux,
)
from src.graphic.model import BroadcastGraphicVerifier
from src.roi.crop import RefereeTubeletExtractor, crop_with_boxes
from src.roi.model import RefereeRoIVerifier
from src.roi.rerank import (
    apply_banner_ownership,
    build_feature_matrix,
    extract_centered_v2_raw,
    score_candidates,
    score_graphic_strips,
)
from src.v2.model import load_model_v2


def build_match_candidates(dets_v2, dets_tdeed, fuse_cfg, top_k=55):
    """Build Stage-1 candidate dicts for one full broadcast match."""
    p_v2_cfg, p_td_cfg = fuse_cfg["platt"][0], fuse_cfg["platt"][1]
    v2_list = [(float(t), float(s), float(apply_platt(p_v2_cfg, s))) for t, s in dets_v2]
    td_list = [(float(t), float(s), float(apply_platt(p_td_cfg, s))) for t, s in dets_tdeed]

    fused = fuse_streams(dets_v2, dets_tdeed, fuse_cfg)
    # Sort by raw fused score descending
    fused_sorted = sorted(fused, key=lambda x: -x[2])

    chosen = []
    seen_t = []
    for t, s_cal, s_raw in fused_sorted:
        if len(chosen) < top_k or s_cal >= 0.12:
            chosen.append((t, s_cal, s_raw))
            seen_t.append(t)

    cands = []
    for t, s_cal, s_raw in chosen:
        v2_25 = [x for x in v2_list if abs(x[0] - t) <= 2.5]
        v2_50 = [x for x in v2_list if abs(x[0] - t) <= 5.0]
        td_25 = [x for x in td_list if abs(x[0] - t) <= 2.5]
        td_50 = [x for x in td_list if abs(x[0] - t) <= 5.0]

        s_v2_raw = max((x[1] for x in v2_25), default=0.0)
        p_v2_val = max((x[2] for x in v2_25), default=0.0)
        p_v2_50 = max((x[2] for x in v2_50), default=0.0)

        s_td_raw = max((x[1] for x in td_25), default=0.0)
        p_td_val = max((x[2] for x in td_25), default=0.0)
        p_td_50 = max((x[2] for x in td_50), default=0.0)

        cands.append({
            "t": float(t),
            "s_cal_s1": float(s_cal),
            "s_fused": float(s_raw),
            "s_v2_raw": float(s_v2_raw),
            "p_v2": float(p_v2_val),
            "p_v2_50": float(p_v2_50),
            "s_td_raw": float(s_td_raw),
            "p_td": float(p_td_val),
            "p_td_50": float(p_td_50),
            "center_offset": 12,
        })
    return cands, fused_sorted


def _decode_match_candidates_worker(task):
    """Worker: opens one match MP4 and extracts 25-frame RoI clips + 18-step graphic strips for all candidates."""
    video_path, cands = task
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    dur = float(stream.duration * tb) if stream.duration else float(stream.frames or 217500) / 25.0

    roi_clips = []
    graf_records = []

    for c in cands:
        t_cand = float(c["t"])
        # Target timestamps:
        # [0]: f_ref at t_cand - 3.0s
        # [1..25]: 25 RoI frames at t_cand - 2.4 + 0.2*i (i=0..24)
        # [26..61]: 18 pairs (t_cand + off, t_cand + off + 0.4)
        targets = [max(0.0, min(dur - 0.1, t_cand + REF_OFFSET_SEC))]
        for i in range(25):
            targets.append(max(0.0, min(dur - 0.1, t_cand - 2.4 + 0.2 * i)))
        for off in STEP_OFFSETS_SEC:
            targets.append(max(0.0, min(dur - 0.1, t_cand + off)))
            targets.append(max(0.0, min(dur - 0.1, t_cand + off + PAIR_DELTA_SEC)))

        order = sorted(range(len(targets)), key=lambda i: targets[i])
        sorted_t = [targets[i] for i in order]
        out = [None] * len(targets)

        seek_ts = int(max(0.0, sorted_t[0] - 0.6) / tb)
        try:
            container.seek(seek_ts, backward=True, any_frame=False, stream=stream)
            ptr = 0
            for frame in container.decode(stream):
                pts_sec = float(frame.pts * tb) if frame.pts is not None else sorted_t[ptr]
                if pts_sec >= sorted_t[ptr] - (0.5 / fps):
                    arr = frame.reformat(width=398, height=224, format="rgb24").to_ndarray()
                    t_u8 = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                    while ptr < len(sorted_t) and pts_sec >= sorted_t[ptr] - (0.5 / fps):
                        out[order[ptr]] = t_u8
                        ptr += 1
                    if ptr >= len(sorted_t):
                        break
        except Exception:
            pass

        for i in range(len(out)):
            if out[i] is None:
                out[i] = out[i - 1].clone() if i > 0 and out[i - 1] is not None else torch.zeros((3, 224, 398), dtype=torch.uint8)

        f_ref = out[0]
        roi_clip = torch.stack(out[1:26], dim=0)  # (25, 3, 224, 398) uint8
        f1_u8 = torch.stack(out[26::2], dim=0)    # (18, 3, 224, 398) uint8
        f2_u8 = torch.stack(out[27::2], dim=0)    # (18, 3, 224, 398) uint8

        strips_u8, ref_u8, g_aux = compute_strip_and_gaux(f_ref, f1_u8, f2_u8)
        roi_clips.append(roi_clip)
        graf_records.append({
            "strips": strips_u8,
            "ref": ref_u8,
            "g_aux": g_aux,
        })

    container.close()
    return roi_clips, graf_records


def extract_and_cache_stage2(games, cache_dir, fuse_cfg, device, workers=6):
    """Extract and score Stage-2 features for all custom games not yet cached."""
    os.makedirs(cache_dir, exist_ok=True)
    pending = []
    for v_path, j_path in games:
        stem = os.path.splitext(os.path.basename(v_path))[0].replace(" ", "_")
        s2_path = os.path.join(cache_dir, f"{stem}_stage2.pt")
        if not os.path.exists(s2_path):
            peaks_path = os.path.join(cache_dir, f"{stem}_peaks.pt")
            cdata = torch.load(peaks_path, map_location="cpu", weights_only=False)
            cands, _ = build_match_candidates(cdata["dets_v2"], cdata["dets_tdeed"], fuse_cfg)
            for c in cands:
                c["game"] = stem
                c["half"] = 1
            pending.append((stem, v_path, cands, s2_path))

    if not pending:
        print("All 11 matches already have cached Stage-2 scores.")
        return

    print(f"Decoding Stage-2 windows & overlay strips for {len(pending)} matches ({sum(len(x[2]) for x in pending)} candidates) across {workers} workers...")
    t0 = time.time()
    decoded_results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_decode_match_candidates_worker, (item[1], item[2])) for item in pending]
        for fut in futs:
            decoded_results.append(fut.result())
    print(f"Decoded all candidate clips & graphic strips in {time.time() - t0:.1f}s.")

    # Load models once on GPU
    print("Loading Stage-2 models (CardNetV2 + FasterRCNN + RefereeRoIVerifier + BroadcastGraphicVerifier)...")
    v2_model, _, _ = load_model_v2("runs/v2_r2/best.pt", device)
    extractor = RefereeTubeletExtractor(v2_model, device, crop_size=128, kf_stride=8)

    roi_ckpt = torch.load("runs/roi_v1/best.pt", map_location="cpu", weights_only=False)
    roi_model = RefereeRoIVerifier(pretrained=False).to(device).eval()
    roi_model.load_state_dict(roi_ckpt["model"])

    graf_ckpt = torch.load("runs/graphic_v1/best.pt", map_location="cpu", weights_only=False)
    graf_model = BroadcastGraphicVerifier(pretrained=False).to(device).eval()
    graf_model.load_state_dict(graf_ckpt["model"])

    for (stem, v_path, cands, s2_path), (roi_clips, graf_records) in zip(pending, decoded_results):
        t_m = time.time()
        B = 24
        for s in range(0, len(cands), B):
            sub_clips = torch.stack(roi_clips[s:s + B])
            boxes, aux = extractor.compute_trajectories(sub_clips)
            crops = crop_with_boxes(sub_clips.to(device), boxes, extractor.crop_size).cpu()
            for k in range(len(sub_clips)):
                cands[s + k]["boxes"] = boxes[k]
                cands[s + k]["aux"] = aux[k]
                cands[s + k]["crop"] = crops[k]

        roi_raw = score_candidates(roi_model, cands, device, batch_size=64, tta_flip=True)
        v2c_raw = extract_centered_v2_raw(cands)
        g_out, g_steps = score_graphic_strips(graf_model, graf_records, device, batch_size=64, tta_flip=True)

        # Save compact cache (without storing full uint8 crops to keep cache small & fast, or store g_aux)
        for k, c in enumerate(cands):
            c.pop("crop", None)
            c["g_aux"] = graf_records[k]["g_aux"]

        torch.save({
            "stem": stem,
            "cands": cands,
            "roi_raw": roi_raw,
            "v2c_raw": v2c_raw,
            "g_out": g_out,
            "g_steps": g_steps,
        }, s2_path)
        print(f"  Cached {stem[-25:]} ({len(cands)} cands) in {time.time() - t_m:.1f}s -> {s2_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", default="custom_data")
    p.add_argument("--cache-dir", default="custom_data/cache_peaks")
    p.add_argument("--fuse-cfg", default="runs/ens_r3_best/bench/fuse.json")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.fuse_cfg) as f:
        fuse_cfg = json.load(f)
    games = find_well_annotated_games(args.batch)
    extract_and_cache_stage2(games, args.cache_dir, fuse_cfg, device, workers=args.workers)
