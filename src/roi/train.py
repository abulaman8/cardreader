"""Train the Stage-2 Referee RoI Tubelet Verifier on cached referee trajectories.

Usage:
  venv/bin/python -m src.roi.train --name roi_v1
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from benchmark.benchmark import SPLIT_PATH, average_precision, match
from ensemble.fuse import val_labels
from src.dataset import gaussian_targets
from src.roi.crop import crop_with_boxes
from src.roi.model import RefereeRoIVerifier
from src.v2.common import get_device
from src.v2.frames import JpegSeq


def _cpu_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


class EMA:
    def __init__(self, model, decay=0.996):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.pairs = list(zip(self.model.state_dict().values(), model.state_dict().values()))

    @torch.no_grad()
    def update(self, model=None):
        d = self.decay
        for pe, pm in self.pairs:
            if pe.dtype.is_floating_point:
                pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
            else:
                pe.copy_(pm)


def _worker_init(_):
    torch.set_num_threads(1)


class RoITrainDataset(Dataset):
    def __init__(self, records, T=25, margin=5, neg_per_pos=2.5, mined_frac=0.65, seed=0):
        self.T = T
        self.margin = margin
        self.neg_per_pos = neg_per_pos
        self.mined_frac = mined_frac
        self.seed = seed

        self.pos = [r for r in records if r["kind"] == "pos" and os.path.exists(r["path"])]
        self.mined = [r for r in records if r["kind"] == "mined" and os.path.exists(r["path"])]
        self.neg = [r for r in records if r["kind"] == "neg" and os.path.exists(r["path"])]
        self.resample(0)

    def resample(self, epoch):
        rng = np.random.default_rng((self.seed, epoch))
        n_neg = int(round(self.neg_per_pos * len(self.pos)))
        n_mined = min(len(self.mined), int(round(self.mined_frac * n_neg)))
        n_rule = min(len(self.neg), n_neg - n_mined)

        items = [("pos", i) for i in range(len(self.pos))]
        if n_mined > 0:
            items += [("mined", int(i)) for i in rng.choice(len(self.mined), n_mined, replace=False)]
        if n_rule > 0:
            items += [("neg", int(i)) for i in rng.choice(len(self.neg), n_rule, replace=False)]
        rng.shuffle(items)
        self.items = items
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, j = self.items[idx]
        rng = np.random.default_rng((self.seed, self.epoch, idx, 17))
        T, m = self.T, self.margin

        if kind == "pos":
            rec = self.pos[j]
            L = len(rec["boxes"])
            p = int(rng.integers(max(0, rec["anchor"] - (L - T)), min(T, rec["anchor"] + 1)))
            p = int(np.clip(p, self.margin, T - 1 - self.margin))
            s = int(np.clip(rec["anchor"] - p, 0, L - T))
            cards = np.asarray(rec["cards"], dtype=np.float32) - s
            y_frame = gaussian_targets(T, cards, sigma=3.0)
            y_clip = float(np.any((cards >= 0) & (cards < T)))
        else:
            rec = self.mined[j] if kind == "mined" else self.neg[j]
            L = len(rec["boxes"])
            s = int(rng.integers(0, max(1, L - T + 1)))
            y_frame = np.zeros(T, dtype=np.float32)
            y_clip = 0.0

        off0 = int(rec.get("start_offset", 0))
        clip_u8 = JpegSeq.open_clip(rec["path"]).decode(off0 + s, T)  # (T, 3, H, W)
        boxes = rec["boxes"][s:s + T].float().clone()                 # (T, 3) [cx, cy, sz]
        aux = rec["aux"][s:s + T].float().clone()                     # (T, 4)

        # Box trajectory spatial jitter (same offset for the whole clip tubelet)
        dx = float(rng.uniform(-8.0, 8.0))
        dy = float(rng.uniform(-8.0, 8.0))
        ds = float(rng.uniform(0.90, 1.12))
        boxes[:, 0] += dx
        boxes[:, 1] += dy
        boxes[:, 2] *= ds

        hflip = bool(rng.random() < 0.5)
        gain = float(rng.uniform(0.88, 1.12))

        return (
            clip_u8,
            boxes,
            aux,
            torch.from_numpy(y_frame),
            torch.tensor(y_clip, dtype=torch.float32),
            torch.tensor(hflip, dtype=torch.bool),
            torch.tensor(gain, dtype=torch.float32),
        )


@torch.no_grad()
def evaluate_val_candidates(model, val_cands, gts_val, ign_val, device, batch_size=64):
    from ensemble.fuse import apply_platt, fit_platt

    model.eval()
    probs_clip = []
    probs_peak = []

    for s in range(0, len(val_cands), batch_size):
        sub = val_cands[s:s + batch_size]
        crops = torch.stack([c["crop"] for c in sub]).to(device, non_blocking=True)
        aux = torch.stack([c["aux"].float() for c in sub]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            f_logits, c_logit = model(crops, aux)
        p_c = torch.sigmoid(c_logit.float()).cpu().numpy()
        p_f = torch.sigmoid(f_logits.float()).cpu().numpy()
        for k, c in enumerate(sub):
            probs_clip.append(float(p_c[k]))
            off = c["center_offset"]
            r0, r1 = max(0, off - 4), min(p_f.shape[1], off + 5)
            probs_peak.append(float(p_f[k, r0:r1].max()))

    probs_clip = np.asarray(probs_clip, dtype=np.float64)
    probs_peak = np.asarray(probs_peak, dtype=np.float64)
    p_roi_raw = 0.55 * probs_clip + 0.45 * probs_peak

    # Candidate-level binary labels (within +-5s of GT card) for standalone RoI AUC & AP
    y_cand = np.array(
        [
            1.0 if any(abs(c["t"] - gt_t) <= 5.0 for gt_t in gts_val.get((c["game"], c["half"]), [])) else 0.0
            for c in val_cands
        ],
        dtype=np.float32,
    )
    roi_auc = float(roc_auc_score(y_cand, p_roi_raw))
    roi_ap = float(average_precision_score(y_cand, p_roi_raw))

    # Platt-scale p_roi on Val before blending with Platt-scaled s_fused & p_v2
    val_roi_dets = [((c["game"], c["half"]), c["t"], float(s)) for c, s in zip(val_cands, p_roi_raw)]
    platt_roi = fit_platt(val_roi_dets, gts_val, ign_val, 5.0)
    p_roi = apply_platt(platt_roi, p_roi_raw)

    raw_dets = [((c["game"], c["half"]), c["t"], c["s_fused"]) for c in val_cands]
    _, _, n_gt = match(raw_dets, gts_val, ign_val, 5.0)

    best_ap5 = 0.0
    best_alpha = 0.0
    for alpha in np.linspace(0.05, 0.50, 10):
        dets_blend = []
        for idx, c in enumerate(val_cands):
            rescue = np.sqrt(max(c["p_v2"] * p_roi[idx], 0.0))
            s_comb = (1.0 - alpha) * c["s_fused"] + alpha * (0.55 * p_roi[idx] + 0.45 * rescue)
            dets_blend.append(((c["game"], c["half"]), c["t"], float(s_comb)))
        _, tp_b, _ = match(dets_blend, gts_val, ign_val, 5.0)
        ap5 = average_precision(tp_b, n_gt)
        if ap5 > best_ap5:
            best_ap5 = ap5
            best_alpha = float(alpha)

    return {
        "val_ap5": float(best_ap5),
        "roi_auc": roi_auc,
        "roi_ap": roi_ap,
        "score": float(best_ap5 + 0.08 * roi_ap),
        "best_alpha": best_alpha,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="roi_v1")
    p.add_argument("--init-v2", default="runs/v2_r2/best.pt")
    p.add_argument("--train-boxes", default="data/roi/train_boxes.pt")
    p.add_argument("--eval-cands", default="data/roi/eval_cands.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr-backbone", type=float, default=2.0e-5)
    p.add_argument("--lr-head", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device(args.device)

    print(f"Loading {args.train_boxes} and {args.eval_cands}...", flush=True)
    records = torch.load(args.train_boxes, map_location="cpu", weights_only=False)
    eval_cands = torch.load(args.eval_cands, map_location="cpu", weights_only=False)
    val_cands = eval_cands["val"]

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    gts_val, ign_val = val_labels("soccernet_data", split["val"])

    ds = RoITrainDataset(records, T=25, margin=5, neg_per_pos=1.8, mined_frac=0.65, seed=0)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        worker_init_fn=_worker_init,
        drop_last=True,
    )

    model = RefereeRoIVerifier(pretrained=False).to(device)
    if args.init_v2 and os.path.exists(args.init_v2):
        model.init_from_v2(args.init_v2)
    ema = EMA(model, decay=0.996)

    # Evaluate Epoch 0 (zero-shot transferred v2_r2 on 128x128 referee crops)
    m0 = evaluate_val_candidates(model, val_cands, gts_val, ign_val, device)
    best_score = m0["score"]
    torch.save(
        {
            "epoch": 0,
            "val_ap5": m0["val_ap5"],
            "roi_auc": m0["roi_auc"],
            "roi_ap": m0["roi_ap"],
            "best_alpha": m0["best_alpha"],
            "model": _cpu_state(model.state_dict()),
        },
        os.path.join(out_dir, "best.pt"),
    )
    print(
        f"Epoch 00/{args.epochs:02d} (init) | roi_auc={m0['roi_auc']:.4f} roi_ap={m0['roi_ap']:.4f} | "
        f"val_ap5={m0['val_ap5']:.4f} | alpha={m0['best_alpha']:.2f} --> Saved initial best.pt",
        flush=True,
    )

    bb_ids = {id(p) for p in model.backbone_params()}
    bb_params = [p for p in model.parameters() if id(p) in bb_ids]
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]

    opt = torch.optim.AdamW(
        [{"params": bb_params, "lr": args.lr_backbone}, {"params": head_params, "lr": args.lr_head}],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    warmup_steps = int(0.10 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ds.resample(epoch)
        model.train()
        loss_sum = 0.0

        for clips_u8, boxes, aux, y_frame, y_clip, hflip, gain in loader:
            clips_u8 = clips_u8.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            y_frame = y_frame.to(device, non_blocking=True)
            y_clip = y_clip.to(device, non_blocking=True)
            hflip = hflip.to(device, non_blocking=True)
            gain = gain.to(device, non_blocking=True)

            with torch.no_grad():
                crops = crop_with_boxes(clips_u8, boxes, crop_size=128)
                if hflip.any():
                    crops[hflip] = crops[hflip].flip(-1)
                # Full brightness + contrast + saturation color jitter on GPU
                f_c = crops.float().div_(255.0)
                f_c = f_c * gain[:, None, None, None, None]
                c_jit = torch.empty((f_c.shape[0], 1, 1, 1, 1), device=device).uniform_(0.82, 1.18)
                s_jit = torch.empty((f_c.shape[0], 1, 1, 1, 1), device=device).uniform_(0.82, 1.18)
                mean_c = f_c.mean(dim=(2, 3, 4), keepdim=True)
                f_c = (f_c - mean_c) * c_jit + mean_c
                gray_c = f_c.mean(dim=2, keepdim=True)
                f_c = (f_c - gray_c) * s_jit + gray_c
                crops = f_c.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                f_logits, c_logit = model(crops, aux)
                loss_f = F.binary_cross_entropy_with_logits(f_logits, y_frame)
                loss_c = F.binary_cross_entropy_with_logits(c_logit, y_clip)
                loss = loss_f + 0.50 * loss_c

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)

            loss_sum += float(loss.item())
            step += 1

        avg_loss = loss_sum / len(loader)
        metrics_raw = evaluate_val_candidates(model, val_cands, gts_val, ign_val, device)
        metrics_ema = evaluate_val_candidates(ema.model, val_cands, gts_val, ign_val, device)

        use_ema = metrics_ema["score"] >= metrics_raw["score"]
        chosen_model = ema.model if use_ema else model
        chosen_metrics = metrics_ema if use_ema else metrics_raw

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | loss={avg_loss:.4f} | "
            f"roi_auc={chosen_metrics['roi_auc']:.4f} roi_ap={chosen_metrics['roi_ap']:.4f} | "
            f"val_ap5={chosen_metrics['val_ap5']:.4f} (raw={metrics_raw['val_ap5']:.4f}, ema={metrics_ema['val_ap5']:.4f}) | "
            f"alpha={chosen_metrics['best_alpha']:.2f}",
            flush=True,
        )

        if chosen_metrics["score"] > best_score:
            best_score = chosen_metrics["score"]
            ckpt = {
                "epoch": epoch,
                "val_ap5": chosen_metrics["val_ap5"],
                "roi_auc": chosen_metrics["roi_auc"],
                "roi_ap": chosen_metrics["roi_ap"],
                "best_alpha": chosen_metrics["best_alpha"],
                "model": _cpu_state(chosen_model.state_dict()),
            }
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            print(
                f"  --> Saved best.pt (val_ap5={chosen_metrics['val_ap5']:.4f}, roi_ap={chosen_metrics['roi_ap']:.4f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
