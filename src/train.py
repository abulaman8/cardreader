"""Train CardTransformer on cached train-game features; validate on the 20 held-out val games every epoch.

Every epoch: runs/<name>/epoch_XXX.pt (model + config + metrics), metrics appended to log.jsonl,
summary in train.log, scalars in tensorboard. best.pt = best val select_metric. last.pt = resumable state.

  python -m src.train --name v1
  python -m src.train --name t40 --set data.window=40 data.margin=12
  python -m src.train --name v1 --resume runs/v1/last.pt
"""
import argparse
import json
import logging
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .common import get_device, load_index
from .config import add_config_args, load_config
from .dataset import EvalClips, TrainClips
from .evaluate import evaluate_split, flatten_metrics
from .model import build_model

NO_DECAY = ("cls", "mask_token", "pos")


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n in NO_DECAY else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def setup_logger(path):
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(path)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _num(x):
    return float("-inf") if x is None or (isinstance(x, float) and math.isnan(x)) else x


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--name", default=time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="path to last.pt")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg["train"]
    device = get_device(args.device)
    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])

    run_dir = os.path.join(cfg["paths"]["runs"], args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log = setup_logger(os.path.join(run_dir, "train.log"))

    index = load_index(cfg)
    train_ds = TrainClips(cfg, index)
    val_clips = EvalClips(cfg, index, "val")
    log.info(f"train items/epoch {len(train_ds)} (pos {len(train_ds.pos)}, neg pool {len(train_ds.neg)}) | "
             f"val clips {len(val_clips)} | device {device}")

    model = build_model(cfg).to(device)
    log.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"])
    steps_per_epoch = math.ceil(len(train_ds) / tc["batch_size"])
    total = steps_per_epoch * tc["epochs"]
    warm = max(1, int(tc["warmup"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cw = cfg["targets"]["clip_weight"]
    select = tc["select_metric"]

    start_epoch, best, bad = 1, float("-inf"), 0
    if args.resume:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
        start_epoch, best, bad = st["epoch"] + 1, st["best"], st["bad"]
        log.info(f"resumed from {args.resume} at epoch {start_epoch}")

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    for epoch in range(start_epoch, tc["epochs"] + 1):
        t0 = time.time()
        train_ds.resample(epoch)
        loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"],
                            pin_memory=device.type == "cuda", drop_last=False,
                            generator=torch.Generator().manual_seed(cfg["data"]["seed"] * 1000 + epoch))
        model.train()
        sums, n = np.zeros(3), 0
        for x, y, c in loader:
            x, y, c = x.to(device, non_blocking=True), y.to(device, non_blocking=True), c.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                fl, cl = model(x)
            lf = F.binary_cross_entropy_with_logits(fl.float(), y)
            lc = F.binary_cross_entropy_with_logits(cl.float(), c)
            loss = lf + cw * lc
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            b = x.shape[0]
            sums += np.array([loss.item(), lf.item(), lc.item()]) * b
            n += b
        train_time = time.time() - t0

        val = evaluate_split(model, cfg, index, "val", device, clips=val_clips)
        metrics = {"epoch": epoch, "train/loss": sums[0] / n, "train/loss_frame": sums[1] / n,
                   "train/loss_clip": sums[2] / n, "lr": sched.get_last_lr()[0], "time/train_s": train_time,
                   "time/epoch_s": time.time() - t0, **flatten_metrics(val, "val")}

        ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg, "metrics": metrics,
                    "val_grid": val.get("grid", [])}, ckpt_path)
        score = _num(val.get(select))
        improved = score > best
        if improved:
            best, bad = score, 0
            shutil.copyfile(ckpt_path, os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        metrics.update({"ckpt": ckpt_path, "best_so_far": best, "is_best": improved})
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "config": cfg,
                    "best": best, "bad": bad}, os.path.join(run_dir, "last.pt"))

        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                writer.add_scalar(k, v, epoch)
        writer.flush()
        g = lambda k: metrics.get(f"val/{k}", float("nan"))
        log.info(f"ep {epoch:03d} | loss {metrics['train/loss']:.4f} | val loss {g('loss'):.4f} "
                 f"clip_auc {g('clip_auc'):.4f} clip_ap {g('clip_ap'):.4f} | tight {g('tight_mAP'):.4f} "
                 f"loose {g('loose_mAP'):.4f} | best thr {g('best_thr')} F1 {g('best_f1'):.4f} "
                 f"(P {g('best_precision'):.3f} R {g('best_recall'):.3f}) | {metrics['time/epoch_s']:.0f}s"
                 f"{' *' if improved else ''}")
        if tc["patience"] and bad >= tc["patience"]:
            log.info(f"early stop: no {select} improvement for {bad} epochs")
            break
    writer.close()
    log.info(f"done. best val {select} = {best:.4f}")


if __name__ == "__main__":
    main()
