"""Train the T-DEED replica (binary card spotting) on extracted train windows.

  python -m tdeed.train --name tdeed_r0
  python -m tdeed.train --name tdeed_r1 --init runs/tdeed_r0/best.pt --mined data/tdeed/mined_round1.json
  python -m tdeed.train --name tdeed_r0 --resume runs/tdeed_r0/last.pt

Every epoch: EMA checkpoint runs/<name>/epoch_XXX.pt, last.pt (resume state), log.jsonl, train.log, tensorboard.
Val clip loss every epoch; full-half spotting on train.spot_subset every train.spot_every epochs and the last
epoch; best.pt = best spot AP@+-5 s; early stop after train.patience spotting evals without improvement.
"""
import argparse
import faulthandler
import gc
import json
import logging
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.hooks
from torch.utils.hooks import warn_if_has_hooks
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .augment import augment_batch, mixup
from .common import add_config_args, decode_batch, get_device, gt_dicts, keys_for_games, load_config, load_index
from .data import TrainWindows, ValClips, collate
from .infer import run_spotting
from .model import build_model, vote
from .spotting import spot_metrics

faulthandler.enable()


class EMA:
    """EMA of all floating state; tensor pairs cached once (params/buffers are updated in place)."""

    def __init__(self, model, decay):
        import copy
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.pairs = list(zip(self.model.state_dict().values(), model.state_dict().values()))

    @torch.no_grad()
    def update(self):
        for e, m in self.pairs:
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def param_groups(model, tc):
    bb = {id(p) for p in model.backbone_params()}
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = p.ndim >= 2 and "temp_enc" not in n
        groups.setdefault((id(p) in bb, decay), []).append(p)
    return [{"params": ps, "lr": tc["lr_backbone"] if is_bb else tc["lr_head"],
             "weight_decay": tc["weight_decay"] if decay else 0.0} for (is_bb, decay), ps in groups.items()]


def lr_lambda(warm, total):
    def f(s):
        if s < warm:
            return 0.01 + 0.99 * s / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    return f


def losses(logits, disp, y, dt, m, cfg):
    """Weighted soft-target CE (bg weight 1, card weight fg_weight) + masked displacement MSE."""
    tc = cfg["targets"]
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(y * logp).sum(-1)                                                    # (B, T)
    cw = torch.tensor([1.0, float(tc["fg_weight"])], device=logits.device)
    w = (y * cw).sum(-1) * m
    l_cls = (w * ce).sum() / w.sum().clamp_min(1e-6)
    l_disp = (((disp.float() - dt) ** 2) * m).sum() / m.sum().clamp_min(1.0)
    return l_cls + tc["disp_weight"] * l_disp, l_cls, l_disp


def to_gpu(batch, device, cfg, train, rng):
    bufs, cls, dt, m = batch
    x = decode_batch(bufs, device).float().div_(255.0)                          # (B, T, 3, H, W)
    cls, dt, m = cls.to(device), dt.to(device), m.to(device)
    y = torch.stack([1 - cls, cls], dim=-1)                                     # soft targets (B, T, 2)
    if train:
        a = cfg["augment"]
        x = augment_batch(x, rng, a)
        if a["mixup"] and x.shape[0] > 1:
            x, y, dt, m = mixup(x, y, dt, m, a["mixup_alpha"], rng)
    return x, y, dt, m


def _safe(fn, y, s):
    return float(fn(y, s)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def val_eval(model, ds, cfg, device):
    model.eval()
    amp = device.type == "cuda"
    tot, n, has, score = np.zeros(3), 0, [], []
    for batch in DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=0, collate_fn=collate):
        x, y, dt, m = to_gpu(batch, device, cfg, False, None)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits, disp = model(x)
        ls = losses(logits, disp, y, dt, m, cfg)
        tot += np.array([v.item() for v in ls]) * x.shape[0]
        n += x.shape[0]
        p = vote(torch.softmax(logits.float(), -1)[..., 1], disp.float())
        score.append(p.max(1).values.cpu().numpy())
        has.append((y[..., 1].max(1).values > 0.5).float().cpu().numpy())
    has, score = np.concatenate(has), np.concatenate(score)
    return {"loss": tot[0] / n, "loss_cls": tot[1] / n, "loss_disp": tot[2] / n,
            "clip_auc": _safe(roc_auc_score, has, score), "clip_ap": _safe(average_precision_score, has, score),
            "n_clips": int(n)}


def setup_logger(path):
    log = logging.getLogger("tdeed_train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(path)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return x


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(_to_cpu(obj), tmp)
    os.replace(tmp, path)


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--name", default=time.strftime("tdeed_%Y%m%d_%H%M%S"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--init", default=None, help="start from a checkpoint's (EMA) weights")
    p.add_argument("--mined", nargs="*", default=[], help="mined window spec json files (extracted with --kind mined)")
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    if args.resume:
        cfg = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
    tc = cfg["train"]
    device = get_device(args.device)
    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])
    torch.set_num_threads(4)

    run_dir = os.path.join(cfg["paths"]["runs"], args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "_cli": {"init": args.init, "mined": args.mined}}, f, sort_keys=False)
    log = setup_logger(os.path.join(run_dir, "train.log"))

    index = load_index(cfg)
    mined = []
    for path in args.mined:
        with open(path) as f:
            mined += json.load(f)
    train_ds = TrainWindows(cfg, index, mined)
    val_ds = ValClips(cfg, index, "val")
    spot_keys = keys_for_games(index["splits"][tc["spot_subset"]])
    gts, ign = gt_dicts(index, spot_keys)
    log.info(f"train/epoch {len(train_ds)} {train_ds.counts} | val clips {len(val_ds)} | "
             f"spot {tc['spot_subset']} ({len(spot_keys)} halves) | device {device}")

    model = build_model(cfg).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        log.info(f"initialised from {args.init}")
    n_bb = sum(p.numel() for p in model.backbone_params())
    log.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (backbone {n_bb / 1e6:.2f}M)")
    ema = EMA(model, tc["ema_decay"])
    use_fused = (device.type == "cuda")
    opt = torch.optim.AdamW(param_groups(model, tc), fused=use_fused)
    acc = max(1, int(tc["acc_grad"]))
    steps = (len(train_ds) // tc["batch_size"]) // acc
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(max(1, tc["warmup_epochs"] * steps), steps * tc["epochs"]))
    amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start, best, bad = 1, float("-inf"), 0
    if args.resume:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_raw"])
        ema.model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        if use_fused:
            for g in opt.param_groups:
                g["fused"] = True
            for p, s in opt.state.items():
                for k, v in s.items():
                    if torch.is_tensor(v) and v.device != p.device:
                        s[k] = v.to(p.device)
        start, best, bad = st["epoch"] + 1, st["best"], st["bad"]
        log.info(f"resumed at epoch {start}")

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    for epoch in range(start, tc["epochs"] + 1):
        t0 = time.time()
        train_ds.resample(epoch)
        loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"],
                            collate_fn=collate, drop_last=True, pin_memory=False, persistent_workers=False,
                            generator=torch.Generator().manual_seed(cfg["data"]["seed"] * 1000 + epoch))
        rng = np.random.default_rng((cfg["data"]["seed"], epoch, 7))
        model.train()
        opt.zero_grad(set_to_none=True)
        sums, n, nb = np.zeros(3), 0, len(loader)
        for step, batch in enumerate(loader):
            x, y, dt, m = to_gpu(batch, device, cfg, True, rng)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, disp = model(x)
            ls = losses(logits, disp, y, dt, m, cfg)
            scaler.scale(ls[0] / acc).backward()
            if (step + 1) % acc == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:                      # skip schedule on overflow steps
                    sched.step()
                ema.update()
            sums += np.array([v.item() for v in ls]) * x.shape[0]
            n += x.shape[0]
            if (step + 1) % tc["log_every"] == 0 or step + 1 == nb:
                log.info(f"ep {epoch:03d} [{step + 1:04d}/{nb}] loss {sums[0] / n:.4f} "
                         f"(cls {sums[1] / n:.4f} disp {sums[2] / n:.4f}) lr {opt.param_groups[0]['lr']:.2e}")
        train_s = time.time() - t0

        metrics = {"epoch": epoch, "train/loss": sums[0] / n, "train/loss_cls": sums[1] / n,
                   "train/loss_disp": sums[2] / n, "lr_backbone": opt.param_groups[0]["lr"], "time/train_s": train_s}
        log.info(f"ep {epoch:03d} val clips ...")
        metrics.update({f"val/{k}": v for k, v in val_eval(ema.model, val_ds, cfg, device).items()})
        spotted = epoch % tc["spot_every"] == 0 or epoch == tc["epochs"]
        if spotted:
            log.info(f"ep {epoch:03d} spotting {len(spot_keys)} halves ...")
            dets = run_spotting(ema.model, cfg, "val", spot_keys, device, progress=False)
            metrics.update({f"spot/{k}": v for k, v in spot_metrics(dets, gts, ign, cfg).items()})
        metrics["time/epoch_s"] = time.time() - t0

        ckpt = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        _atomic_save({"epoch": epoch, "model": ema.model.state_dict(), "config": cfg, "metrics": metrics}, ckpt)
        improved = False
        if spotted:
            score = metrics.get(f"spot/{tc['select_metric']}")
            score = float("-inf") if score is None or math.isnan(score) else score
            if score > best:
                best, bad, improved = score, 0, True
                shutil.copyfile(ckpt, os.path.join(run_dir, "best.pt"))
            else:
                bad += 1
        metrics.update({"ckpt": ckpt, "best_so_far": best, "is_best": improved})
        _atomic_save({"epoch": epoch, "model": ema.model.state_dict(), "model_raw": model.state_dict(),
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "config": cfg, "best": best, "bad": bad}, os.path.join(run_dir, "last.pt"))
        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                writer.add_scalar(k, v, epoch)
        writer.flush()
        g = lambda k: metrics.get(k, float("nan"))  # noqa: E731
        msg = (f"ep {epoch:03d} | loss {g('train/loss'):.4f} | val loss {g('val/loss'):.4f} "
               f"clip_auc {g('val/clip_auc'):.4f} clip_ap {g('val/clip_ap'):.4f}")
        if spotted:
            msg += (f" | spot AP5 {g('spot/ap5'):.4f} loose {g('spot/loose_mAP'):.4f} F1@{g('spot/best_thr')} "
                    f"{g('spot/best_f1'):.4f} (P {g('spot/best_precision'):.3f} R {g('spot/best_recall'):.3f})")
        log.info(msg + f" | {g('time/epoch_s'):.0f}s{' *' if improved else ''}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if spotted and tc["patience"] and bad >= tc["patience"]:
            log.info(f"early stop: {bad} spotting evaluations without {tc['select_metric']} improvement")
            break
    writer.close()
    log.info(f"done. best spot/{tc['select_metric']} = {best:.4f}")


if __name__ == "__main__":
    main()
