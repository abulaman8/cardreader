"""Referee / Card Region-of-Interest (RoI) Tubelet Extractor & Trajectory Cache.

Uses a pretrained person detector (Faster R-CNN MobileNetV3-Large FPN) scored by
CardNetV2's 4-head spatial attention map + card-color saliency to localize the
referee's upper body and raised hand across a clip, with smooth temporal
interpolation and attention-peak fallback when crowded/occluded.

Usage:
  venv/bin/python -m src.roi.crop --ckpt-v2 runs/v2_r2/best.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.detection as tvd
from tqdm import tqdm

from benchmark.benchmark import SPLIT_PATH, load_preds
from ensemble.fuse import apply_platt, fit_platt, read_run, val_labels
from src.v2.common import clip_path, get_device, half_prefix, load_config, load_index
from src.v2.frames import JpegSeq
from src.v2.model import load_model_v2


def card_color_saliency(frames_f):
    """Compute yellow/red card color saliency on float RGB frames in [0, 1].
    frames_f: (N, 3, H, W) -> returns (N, H, W) in [0, 1].
    """
    r, g, b = frames_f[:, 0], frames_f[:, 1], frames_f[:, 2]
    yellow = torch.clamp(torch.minimum(r, g) - 1.25 * b, min=0.0) * ((r > 0.60) & (g > 0.50)).float()
    red = torch.clamp(r - 1.35 * torch.maximum(g, b), min=0.0) * (r > 0.60).float()
    return torch.clamp(yellow + red, 0.0, 1.0)


def crop_with_boxes(clips_u8, boxes, crop_size=128):
    """Fast GPU affine crop from (B, T, 3, H, W) uint8 using boxes (B, T, 3) [cx, cy, sz]."""
    B, T, C, H, W = clips_u8.shape
    device = clips_u8.device
    boxes = boxes.to(device, dtype=torch.float32)
    cx, cy, sz = boxes[..., 0], boxes[..., 1], boxes[..., 2]
    half = sz / 2.0
    cx_clamped = torch.minimum(torch.maximum(cx, half), W - half)
    cy_clamped = torch.minimum(torch.maximum(cy, half), H - half)

    sx = sz / W
    sy = sz / H
    tx = (2.0 * cx_clamped / W) - 1.0
    ty = (2.0 * cy_clamped / H) - 1.0

    N = B * T
    theta = torch.zeros((N, 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] = sx.reshape(N)
    theta[:, 1, 1] = sy.reshape(N)
    theta[:, 0, 2] = tx.reshape(N)
    theta[:, 1, 2] = ty.reshape(N)

    grid = F.affine_grid(theta, size=(N, C, crop_size, crop_size), align_corners=False)
    crops_f = F.grid_sample(
        clips_u8.flatten(0, 1).float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return crops_f.clamp_(0, 255).round_().to(torch.uint8).view(B, T, C, crop_size, crop_size)


class RefereeTubeletExtractor:
    def __init__(self, v2_model, device, crop_size=128, kf_stride=6, det_thresh=0.35):
        self.v2_model = v2_model.eval()
        self.device = device
        self.crop_size = crop_size
        self.kf_stride = kf_stride
        self.det_thresh = det_thresh
        self.detector = tvd.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT").to(device).eval()
        self.detector.transform.min_size = (320,)
        self.detector.transform.max_size = 640
        for p in self.detector.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_trajectories(self, clips_u8):
        """Compute smooth referee box trajectories (B, T, 3) [cx, cy, sz] and aux features (B, T, 4)
        [v2_conf, v2_disp, det_score, card_sal] from (B, T, 3, H, W) uint8 clips.
        """
        B, T, C, H, W = clips_u8.shape
        x = clips_u8.to(self.device, non_blocking=True)

        # 1. CardNetV2 forward for spatial attention + frame confidence/displacement
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
            v2_conf, v2_disp, _, att = self.v2_model(x, return_attn=True)
        v2_prob = torch.sigmoid(v2_conf.float())  # (B, T)
        v2_disp = v2_disp.float()                 # (B, T)
        att_mean = att.float().mean(dim=2)        # (B, T, Hf, Wf)
        att_up = F.interpolate(
            att_mean.flatten(0, 1)[:, None], size=(H, W), mode="bilinear", align_corners=False
        ).view(B, T, H, W)
        att_min = att_up.flatten(2).min(dim=-1)[0][:, :, None, None]
        att_max = att_up.flatten(2).max(dim=-1)[0][:, :, None, None]
        att_norm = (att_up - att_min) / (att_max - att_min + 1e-6)

        # 2. Keyframe person detection
        kf_indices = list(range(0, T, self.kf_stride))
        if (T - 1) not in kf_indices:
            kf_indices.append(T - 1)
        K = len(kf_indices)

        kf_frames = x[:, kf_indices].flatten(0, 1).float().div_(255.0)
        sal_kf = card_color_saliency(kf_frames).view(B, K, H, W)

        det_outputs = []
        for s in range(0, len(kf_frames), 32):
            det_outputs.extend(self.detector(list(kf_frames[s:s + 32])))

        kf_cx = torch.zeros((B, K), device=self.device)
        kf_cy = torch.zeros((B, K), device=self.device)
        kf_sz = torch.full((B, K), 128.0, device=self.device)
        kf_det = torch.zeros((B, K), device=self.device)
        kf_sal = torch.zeros((B, K), device=self.device)

        gy = torch.arange(H, device=self.device, dtype=torch.float32)[:, None]
        gx = torch.arange(W, device=self.device, dtype=torch.float32)[None, :]

        for b in range(B):
            for ki, t_idx in enumerate(kf_indices):
                out = det_outputs[b * K + ki]
                att_map = att_norm[b, t_idx]
                sal_map = sal_kf[b, ki]

                mask_top = (att_map >= 0.75).float()
                w_sum = mask_top.sum().clamp_min(1.0)
                fb_cy = (mask_top * gy).sum() / w_sum
                fb_cx = (mask_top * gx).sum() / w_sum

                boxes = out["boxes"]
                labels = out["labels"]
                scores = out["scores"]
                keep = (labels == 1) & (scores >= self.det_thresh)
                if keep.any():
                    boxes = boxes[keep]
                    scores = scores[keep]
                    best_score = -1.0
                    best_box = None
                    best_det_s = 0.0
                    best_sal_s = 0.0
                    for bi in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[bi]
                        bw = x2 - x1
                        bh = y2 - y1
                        if bw < 8 or bh < 14:
                            continue
                        ux1 = int(torch.clamp(x1 - 0.25 * bw, 0, W - 1).item())
                        ux2 = int(torch.clamp(x2 + 0.25 * bw, ux1 + 1, W).item())
                        uy1 = int(torch.clamp(y1 - 0.25 * bh, 0, H - 1).item())
                        uy2 = int(torch.clamp(y1 + 0.70 * bh, uy1 + 1, H).item())

                        att_val = att_map[uy1:uy2, ux1:ux2].mean().item()
                        sal_val = sal_map[uy1:uy2, ux1:ux2].mean().item()
                        comb = (att_val + 0.15) * (1.0 + 4.0 * sal_val) * (scores[bi].item() ** 0.35)
                        if comb > best_score:
                            best_score = comb
                            best_box = (x1, y1, x2, y2, bw, bh)
                            best_det_s = scores[bi].item()
                            best_sal_s = sal_val

                    if best_box is not None:
                        x1, y1, x2, y2, bw, bh = best_box
                        kf_cx[b, ki] = 0.5 * (x1 + x2)
                        kf_cy[b, ki] = y1 + 0.38 * bh
                        kf_sz[b, ki] = torch.clamp(torch.maximum(bw * 1.6, bh * 1.15), 96.0, 176.0)
                        kf_det[b, ki] = best_det_s
                        kf_sal[b, ki] = best_sal_s
                        continue

                kf_cx[b, ki] = fb_cx
                kf_cy[b, ki] = fb_cy
                kf_sz[b, ki] = 128.0
                fx1 = int(torch.clamp(fb_cx - 48, 0, W - 1).item())
                fx2 = int(torch.clamp(fb_cx + 48, fx1 + 1, W).item())
                fy1 = int(torch.clamp(fb_cy - 48, 0, H - 1).item())
                fy2 = int(torch.clamp(fb_cy + 48, fy1 + 1, H).item())
                kf_sal[b, ki] = sal_map[fy1:fy2, fx1:fx2].mean().item()

        # 3. Interpolate keyframe trajectories across T frames
        t_grid = torch.arange(T, device=self.device, dtype=torch.float32)
        kf_t = torch.tensor(kf_indices, device=self.device, dtype=torch.float32)

        cx_all = torch.zeros((B, T), device=self.device)
        cy_all = torch.zeros((B, T), device=self.device)
        sz_all = torch.zeros((B, T), device=self.device)
        det_all = torch.zeros((B, T), device=self.device)
        sal_all = torch.zeros((B, T), device=self.device)

        for t_i in range(T):
            t_val = t_grid[t_i]
            right = torch.searchsorted(kf_t, t_val).clamp(0, K - 1)
            left = (right - 1).clamp(0, K - 1)
            denom = (kf_t[right] - kf_t[left]).clamp_min(1e-5)
            w = (t_val - kf_t[left]) / denom
            cx_all[:, t_i] = kf_cx[:, left] * (1.0 - w) + kf_cx[:, right] * w
            cy_all[:, t_i] = kf_cy[:, left] * (1.0 - w) + kf_cy[:, right] * w
            sz_all[:, t_i] = kf_sz[:, left] * (1.0 - w) + kf_sz[:, right] * w
            det_all[:, t_i] = kf_det[:, left] * (1.0 - w) + kf_det[:, right] * w
            sal_all[:, t_i] = kf_sal[:, left] * (1.0 - w) + kf_sal[:, right] * w

        kern = torch.tensor([0.15, 0.20, 0.30, 0.20, 0.15], device=self.device).view(1, 1, 5)
        cx_all = F.conv1d(F.pad(cx_all[:, None], (2, 2), mode="replicate"), kern)[:, 0]
        cy_all = F.conv1d(F.pad(cy_all[:, None], (2, 2), mode="replicate"), kern)[:, 0]
        sz_all = F.conv1d(F.pad(sz_all[:, None], (2, 2), mode="replicate"), kern)[:, 0]

        boxes = torch.stack([cx_all, cy_all, sz_all], dim=-1).cpu().to(torch.float16)
        aux = torch.stack([v2_prob, v2_disp, det_all, sal_all], dim=-1).cpu().to(torch.float16)
        return boxes, aux

    @torch.no_grad()
    def extract_batch(self, clips_u8):
        boxes, _ = self.compute_trajectories(clips_u8)
        return crop_with_boxes(clips_u8.to(self.device), boxes, self.crop_size).cpu()


def collect_eval_candidates(cfg, split_name, games, top_per_half=25):
    """Collect top-K candidates per half from ens_r3_best along with individual v2 and tdeed scores."""
    with open("runs/ens_r3_best/bench/fuse.json") as f:
        fuse_cfg = json.load(f)
    p_v2, p_td = fuse_cfg["platt"][0], fuse_cfg["platt"][1]

    v2_raw = read_run("runs/v2_r2/bench", split_name, games, kind="raw")
    td_raw = read_run("runs/tdeed_r3/bench", split_name, games, kind="raw")
    ens_fused = read_run("runs/ens_r3_best/bench", split_name, games, kind="fused")

    v2_by_gh, td_by_gh, ens_by_gh = {}, {}, {}
    for gh, t, s in v2_raw:
        v2_by_gh.setdefault(gh, []).append((t, s, float(apply_platt(p_v2, s))))
    for gh, t, s in td_raw:
        td_by_gh.setdefault(gh, []).append((t, s, float(apply_platt(p_td, s))))
    for gh, t, s in ens_fused:
        ens_by_gh.setdefault(gh, []).append((t, s))

    cands = []
    for g in games:
        for h in (1, 2):
            gh = (g, h)
            lst = sorted(ens_by_gh.get(gh, []), key=lambda x: -x[1])[:top_per_half]
            v2_list = v2_by_gh.get(gh, [])
            td_list = td_by_gh.get(gh, [])
            for t, s_fused in lst:
                # Find closest v2 and tdeed detections within 2.5s and 5.0s
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
                    "game": g,
                    "half": h,
                    "t": float(t),
                    "s_fused": float(s_fused),
                    "s_v2_raw": float(s_v2_raw),
                    "p_v2": float(p_v2_val),
                    "p_v2_50": float(p_v2_50),
                    "s_td_raw": float(s_td_raw),
                    "p_td": float(p_td_val),
                    "p_td_50": float(p_td_50),
                })
    return cands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/v2.yaml")
    p.add_argument("--ckpt-v2", default="runs/v2_r2/best.pt")
    p.add_argument("--out-dir", default="data/roi")
    p.add_argument("--neg-limit", type=int, default=2500)
    p.add_argument("--top-per-half", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)
    cfg = load_config(args.config)
    index = load_index(cfg)
    v2_model, _, _ = load_model_v2(args.ckpt_v2, device)
    extractor = RefereeTubeletExtractor(v2_model, device, crop_size=128, kf_stride=8)

    L_full = cfg["data"]["cache_len"]  # 45 frames
    L_sub = 31                         # centered 31-frame slice [7:38] allowing +-3 frame jitter for T=25
    off0 = (L_full - L_sub) // 2       # 7
    T = cfg["data"]["window"]          # 25 frames

    # 1. Extract trajectories for training clips (pos + mined + subset of neg)
    train_out = os.path.join(args.out_dir, "train_boxes.pt")
    if not os.path.exists(train_out):
        mined_specs = []
        for mp in ("data/v2/mined_round1.json", "data/v2/mined_round2.json"):
            if os.path.exists(mp):
                with open(mp) as f:
                    mined_specs.extend(json.load(f))

        all_negs = index["train_neg"]
        foul_stop = [c for c in all_negs if c.get("kind") in ("foul", "stoppage")]
        rand_negs = [c for c in all_negs if c.get("kind") not in ("foul", "stoppage")]
        rng = np.random.default_rng(0)
        if len(foul_stop) >= args.neg_limit:
            chosen_negs = [foul_stop[i] for i in rng.choice(len(foul_stop), args.neg_limit, replace=False)]
        else:
            rem = min(len(rand_negs), args.neg_limit - len(foul_stop))
            chosen_negs = foul_stop + [rand_negs[i] for i in rng.choice(len(rand_negs), rem, replace=False)]

        tasks = []
        for c in index["train_pos"]:
            path = clip_path(cfg, "pos", c["id"])
            if os.path.exists(path):
                rel_cards = [float(cf - off0) for cf in c["cards"] if 0 <= (cf - off0) < L_sub]
                tasks.append(("pos", c["id"], path, int(c["anchor"] - off0), rel_cards))
        for c in mined_specs:
            path = clip_path(cfg, "mined", c["id"])
            if os.path.exists(path):
                tasks.append(("mined", c["id"], path, L_sub // 2, []))
        for c in chosen_negs:
            path = clip_path(cfg, "neg", c["id"])
            if os.path.exists(path):
                tasks.append(("neg", c["id"], path, L_sub // 2, []))

        print(f"Extracting referee trajectories for {len(tasks)} training clips (L_sub={L_sub})...")
        records = []
        B = args.batch_size
        for s in tqdm(range(0, len(tasks), B), desc="Train RoI boxes"):
            sub = tasks[s:s + B]
            clips = torch.stack([JpegSeq.open_clip(item[2]).decode(off0, L_sub) for item in sub])
            boxes, aux = extractor.compute_trajectories(clips)
            for k, item in enumerate(sub):
                records.append({
                    "kind": item[0],
                    "id": item[1],
                    "path": item[2],
                    "start_offset": off0,
                    "anchor": item[3],
                    "cards": item[4],
                    "boxes": boxes[k],
                    "aux": aux[k],
                })
        torch.save(records, train_out)
        print(f"Saved {len(records)} training RoI trajectories -> {train_out}")
    else:
        print(f"Found existing {train_out}, skipping train extraction.")

    # 2. Extract trajectories & 128x128 tubelets for Val & Test Stage-1 Candidates
    eval_out = os.path.join(args.out_dir, "eval_cands.pt")
    if not os.path.exists(eval_out):
        with open(SPLIT_PATH) as f:
            split = json.load(f)

        eval_data = {}
        fps = cfg["data"]["fps"]
        for sp_name in ("val", "test"):
            games = split[sp_name]
            cands = collect_eval_candidates(cfg, sp_name, games, top_per_half=args.top_per_half)
            print(f"Extracting {len(cands)} {sp_name} candidate tubelets (T={T})...")

            # Group by (game, half) to open each half's JpegSeq once
            half_cache = {}
            clips_list = []
            for c in cands:
                key = f"{c['game']}|{c['half']}"
                if key not in half_cache:
                    pre = half_prefix(cfg, sp_name, key)
                    half_cache[key] = JpegSeq.open_half(pre)
                seq = half_cache[key]
                cf = int(round(c["t"] * fps))
                start_f = int(np.clip(cf - T // 2, 0, max(len(seq) - T, 0)))
                c["start_frame"] = start_f
                c["center_offset"] = int(np.clip(cf - start_f, 0, T - 1))
                clips_list.append(seq.decode(start_f, T))

            B = args.batch_size
            for s in tqdm(range(0, len(cands), B), desc=f"{sp_name} RoI tubelets"):
                sub_clips = torch.stack(clips_list[s:s + B])
                boxes, aux = extractor.compute_trajectories(sub_clips)
                crops = crop_with_boxes(sub_clips.to(device), boxes, extractor.crop_size).cpu()
                for k in range(len(sub_clips)):
                    cands[s + k]["boxes"] = boxes[k]
                    cands[s + k]["aux"] = aux[k]
                    cands[s + k]["crop"] = crops[k]  # (25, 3, 128, 128) uint8
            eval_data[sp_name] = cands

        torch.save(eval_data, eval_out)
        print(f"Saved Val & Test candidate tubelets -> {eval_out}")
    else:
        print(f"Found existing {eval_out}, skipping eval extraction.")


if __name__ == "__main__":
    main()
