"""Training Script for Stage-2 Post-Event Broadcast Card Graphic Verifier (GraphicNet).

Trains BroadcastGraphicVerifier on data/graphic/train_strips.pt (positives +
Substitution, Goal, Mined, and Foul hard negatives) using Multiple-Instance
Learning (MIL) burst loss + step-level stationary pop-up guidance, evaluating
every epoch on the 2,500 Validation candidates in data/graphic/eval_strips.pt.

Usage:
  venv/bin/python -m src.graphic.train --epochs 14 --out-dir runs/graphic_v1
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from benchmark.benchmark import SPLIT_PATH, match
from ensemble.fuse import val_labels
from src.graphic.model import BroadcastGraphicVerifier
import copy
import random
from src.v2.common import get_device


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ep, mp in zip(self.ema.parameters(), model.parameters()):
            ep.mul_(d).add_(mp.detach(), alpha=1.0 - d)
        for eb, mb in zip(self.ema.buffers(), model.buffers()):
            eb.copy_(mb)



class GraphicStripDataset(Dataset):
    def __init__(self, records, train=True):
        self.records = records
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        strips = r["strips"].clone()          # (18, 3, 96, 256) uint8
        ref = r["ref"].clone()                # (1, 3, 96, 256) uint8
        g_aux = r["g_aux"].float().clone()    # (18, 8) float32
        label = float(r["label"])

        if self.train:
            # 1. Synchronized horizontal flip (keeps strips and ref pixel-aligned)
            if torch.rand(()) < 0.5:
                strips = strips.flip(-1)
                ref = ref.flip(-1)

            # 2. Synchronized slight spatial translation (+-2px vertical, +-6px horizontal)
            if torch.rand(()) < 0.5:
                dy = int(torch.randint(-2, 3, (1,)).item())
                dx = int(torch.randint(-6, 7, (1,)).item())
                strips = torch.roll(strips, shifts=(dy, dx), dims=(-2, -1))
                ref = torch.roll(ref, shifts=(dy, dx), dims=(-2, -1))

            # 3. Synchronized mild brightness/contrast jitter (preserves diff = strips - ref)
            if torch.rand(()) < 0.5:
                gain = float(torch.empty(()).uniform_(0.90, 1.10).item())
                bias = float(torch.empty(()).uniform_(-8.0, 8.0).item())
                strips = (strips.float() * gain + bias).clamp_(0, 255).to(torch.uint8)
                ref = (ref.float() * gain + bias).clamp_(0, 255).to(torch.uint8)

            # 4. Random 1-step temporal roll or dropout for timing invariance
            if torch.rand(()) < 0.35:
                dt = int(torch.randint(-1, 2, (1,)).item())
                if dt != 0:
                    strips = torch.roll(strips, shifts=dt, dims=0)
                    g_aux = torch.roll(g_aux, shifts=dt, dims=0)

        return strips, ref, g_aux, torch.tensor(label, dtype=torch.float32)


def mil_graphic_loss(step_logits, out_logit, g_aux, labels):
    """Multiple-Instance Learning (MIL) loss for post-event broadcast graphics:
      - On negatives (y = 0): all 18 steps, the max step, and out_logit are supervised to 0.
      - On positives (y = 1): out_logit is supervised to a confidence-weighted positive target
        (0.95 when a clear pop-up badge is present, 0.75 soft target when subtle/absent),
        and step_logits receive soft guidance on steps where a pop-up yellow/red badge appears.
    """
    is_pos = (labels > 0.5)
    is_neg = ~is_pos

    # 1. Clip-level MIL burst + CLS loss
    badge_sig = torch.maximum(g_aux[:, :, 0], g_aux[:, :, 2]).amax(dim=1)  # (B,)
    pos_target = torch.where(badge_sig > 0.035, 0.95, 0.75)
    clip_targets = torch.where(is_pos, pos_target, torch.zeros_like(labels))
    loss_clip = F.binary_cross_entropy_with_logits(out_logit, clip_targets)

    # 2. Negative step suppression: every single step of a negative event (incl. Substitutions/Goals) must be 0
    if is_neg.any():
        neg_steps = step_logits[is_neg]
        loss_neg_all = F.binary_cross_entropy_with_logits(neg_steps, torch.zeros_like(neg_steps))
        loss_neg_max = F.binary_cross_entropy_with_logits(
            neg_steps.amax(dim=1), torch.zeros(neg_steps.shape[0], device=step_logits.device)
        )
    else:
        loss_neg_all = step_logits.sum() * 0.0
        loss_neg_max = step_logits.sum() * 0.0

    # 3. Positive step localization guidance using stationary pop-up badge signal
    if is_pos.any():
        pos_steps = step_logits[is_pos]
        pos_g = g_aux[is_pos]
        step_badge = torch.maximum(pos_g[:, :, 0], pos_g[:, :, 2])  # (B_pos, 18)
        # Smooth across adjacent steps (since graphics stay up for 2-3 steps)
        step_badge_sm = F.avg_pool1d(
            F.pad(step_badge.unsqueeze(1), (1, 1), mode="replicate"), kernel_size=3, stride=1
        ).squeeze(1)
        b_max = step_badge_sm.amax(dim=1, keepdim=True).clamp(min=1e-5)
        soft_step_t = (step_badge_sm / b_max).clamp(0.0, 1.0) * (b_max > 0.03).float()
        has_badge = (b_max.squeeze(1) > 0.03)
        if has_badge.any():
            loss_pos_step = F.binary_cross_entropy_with_logits(
                pos_steps[has_badge], soft_step_t[has_badge]
            )
        else:
            loss_pos_step = step_logits.sum() * 0.0
    else:
        loss_pos_step = step_logits.sum() * 0.0

    return loss_clip + 0.45 * loss_neg_all + 0.35 * loss_neg_max + 0.35 * loss_pos_step


@torch.no_grad()
def evaluate_val_candidates(model, val_strips, val_cands, gts_val, ign_val, device, tol=5.0):
    model.eval()
    probs = []
    B = 64
    for s in range(0, len(val_strips), B):
        sub = val_strips[s:s + B]
        strips = torch.stack([x["strips"] for x in sub]).to(device, non_blocking=True)
        ref = torch.stack([x["ref"] for x in sub]).to(device, non_blocking=True)
        g_aux = torch.stack([x["g_aux"].float() for x in sub]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            s_log, o_log = model(strips, ref, g_aux)
            p_o = torch.sigmoid(o_log.float())
            p_s = torch.sigmoid(s_log.float()).amax(dim=1)
            # Horizontal flip TTA
            s_log_f, o_log_f = model(strips.flip(-1), ref.flip(-1), g_aux)
            p_o = 0.5 * (p_o + torch.sigmoid(o_log_f.float()))
            p_s = 0.5 * (p_s + torch.sigmoid(s_log_f.float()).amax(dim=1))
            p = 0.70 * p_o + 0.30 * p_s
        probs.extend(p.cpu().tolist())

    probs = np.asarray(probs, dtype=np.float64)

    # Candidate-level binary labels (within +-5s of GT card) for standalone Graphic AUC & AP
    y_true = np.array(
        [
            1.0 if any(abs(c["t"] - gt_t) <= tol for gt_t in gts_val.get((c["game"], c["half"]), [])) else 0.0
            for c in val_cands
        ],
        dtype=np.float64,
    )

    desc = np.argsort(-probs)
    y_sorted = y_true[desc]
    cum_tp = np.cumsum(y_sorted)
    n_pos = max(float(y_true.sum()), 1.0)
    n_neg = max(float(len(y_true) - y_true.sum()), 1.0)
    prec = cum_tp / np.arange(1, len(y_sorted) + 1)
    graf_ap = float((prec * y_sorted).sum() / n_pos)
    tpr = cum_tp / n_pos
    fpr = np.cumsum(1.0 - y_sorted) / n_neg
    graf_auc = float(np.trapezoid(tpr, fpr))

    return graf_auc, graf_ap, probs



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-strips", default="data/graphic/train_strips.pt")
    p.add_argument("--eval-strips", default="data/graphic/eval_strips.pt")
    p.add_argument("--eval-cands", default="data/roi/eval_cands.pt")
    p.add_argument("--init-v2", default="runs/v2_r2/best.pt")
    p.add_argument("--out-dir", default="runs/graphic_v1")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)

    print(f"Loading {args.train_strips} ...", flush=True)
    train_records = torch.load(args.train_strips, map_location="cpu", weights_only=False)
    eval_strips = torch.load(args.eval_strips, map_location="cpu", weights_only=False)
    eval_cands = torch.load(args.eval_cands, map_location="cpu", weights_only=False)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    gts_val, ign_val = val_labels("soccernet_data", split["val"])

    # Weight sampling so positives (~40%), substitutions/goals (~35%), and mined/foul (~25%) are balanced
    weights = []
    for r in train_records:
        if r["label"] == 1:
            weights.append(2.2)
        elif r["kind"] in ("sub", "goal"):
            weights.append(1.4)
        else:
            weights.append(1.0)

    ds = GraphicStripDataset(train_records, train=True)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = BroadcastGraphicVerifier(pretrained=True).to(device)
    if args.init_v2 and os.path.exists(args.init_v2):
        model.init_from_v2(args.init_v2)
    ema = ModelEMA(model, decay=0.995)

    bb_ids = {id(p) for p in model.backbone_params()}
    bb_params = [p for p in model.parameters() if id(p) in bb_ids and p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in bb_ids and p.requires_grad]

    opt = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps = args.epochs * len(loader)
    warmup_steps = len(loader)
    step_num = 0
    best_score = -1.0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        for strips, ref, g_aux, labels in loader:
            step_num += 1
            if step_num <= warmup_steps:
                scale = step_num / max(1, warmup_steps)
            else:
                prog = (step_num - warmup_steps) / max(1, total_steps - warmup_steps)
                scale = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
            opt.param_groups[0]["lr"] = args.lr * args.backbone_lr_mult * scale
            opt.param_groups[1]["lr"] = args.lr * scale

            strips = strips.to(device, non_blocking=True)
            ref = ref.to(device, non_blocking=True)
            g_aux = g_aux.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                s_log, o_log = model(strips, ref, g_aux)
                loss = mil_graphic_loss(s_log, o_log, g_aux, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            loss_sum += float(loss.item())
            n_batches += 1

        auc_raw, ap_raw, _ = evaluate_val_candidates(
            model, eval_strips["val"], eval_cands["val"], gts_val, ign_val, device
        )
        auc_ema, ap_ema, _ = evaluate_val_candidates(
            ema.ema, eval_strips["val"], eval_cands["val"], gts_val, ign_val, device
        )

        use_ema = (0.5 * auc_ema + 0.5 * ap_ema) >= (0.5 * auc_raw + 0.5 * ap_raw)
        cur_auc = auc_ema if use_ema else auc_raw
        cur_ap = ap_ema if use_ema else ap_raw
        cur_score = 0.5 * cur_auc + 0.5 * cur_ap
        dt = time.time() - t0

        tag = ""
        if cur_score > best_score:
            best_score = cur_score
            best_sd = ema.ema.state_dict() if use_ema else model.state_dict()
            torch.save(
                {
                    "epoch": ep,
                    "model": best_sd,
                    "val_graf_auc": cur_auc,
                    "val_graf_ap": cur_ap,
                },
                os.path.join(args.out_dir, "best.pt"),
            )
            tag = " *best*"

        print(
            f"[Graphic Ep {ep:02d}/{args.epochs:02d}] loss={loss_sum / max(1, n_batches):.4f} | "
            f"val_graf_auc={cur_auc:.4f} val_graf_ap={cur_ap:.4f} ({'EMA' if use_ema else 'RAW'}) | "
            f"{dt:.1f}s{tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
