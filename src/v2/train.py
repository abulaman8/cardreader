"""Train CardNetV2 end-to-end on cached train-game frames.

Every epoch: EMA checkpoint runs/<name>/epoch_XXX.pt (+ metrics), log.jsonl, train.log, tensorboard.
Val clip metrics every epoch; full-half val spotting every train.spot_every epochs (and the last epoch).

  python -m src.v2.train --name v2_r0
  python -m src.v2.train --name v2_r1 --init runs/v2_r0/best.pt --mined data/v2/mined_round1.json
  python -m src.v2.train --name v2_r0 --resume runs/v2_r0/last.pt
"""
import argparse
import copy
import faulthandler
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .common import add_v2_config_args, get_device, gt_dicts, keys_for_games, load_config, load_index, load_split
from .data import TrainClipsV2, ValClipsV2
from .infer import run_spotting, spot_metrics
from .model import build_model_v2

NO_DECAY = ("cls", "pos")
faulthandler.enable()


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.pairs = list(zip(self.model.state_dict().values(), model.state_dict().values()))

    @torch.no_grad()
    def update(self, model=None):
        for e, m in self.pairs:
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def param_groups(model, tc):
    bb = {id(p) for p in model.backbone_params()}
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for n, p in model.named_parameters():
        decay = not (p.ndim < 2 or n in NO_DECAY)
        groups[(id(p) in bb, decay)].append(p)
    return [{"params": ps, "lr": tc["lr_backbone"] if is_bb else tc["lr_head"],
             "weight_decay": tc["weight_decay"] if decay else 0.0}
            for (is_bb, decay), ps in groups.items() if ps]


def losses(conf, disp, clip, y, dt, dm, c, cfg):
    tc = cfg["targets"]
    lf = F.binary_cross_entropy_with_logits(conf.float(), y)
    lc = F.binary_cross_entropy_with_logits(clip.float(), c)
    ld = (F.smooth_l1_loss(disp.float(), dt, reduction="none") * dm).sum() / dm.sum().clamp_min(1.0)
    return lf + tc["clip_weight"] * lc + tc["disp_weight"] * ld, lf, lc, ld


def _safe(fn, y, s):
    return float(fn(y, s)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def clip_eval(model, ds, cfg, device, workers):
    model.eval()
    amp = device.type == "cuda"
    tot, n, cl, cs, fm = np.zeros(4), 0, [], [], []
    for x, y, dt, dm, c in DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=workers):
        x, y, dt, dm, c = (t.to(device, non_blocking=True) for t in (x, y, dt, dm, c))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            conf, disp, clip = model(x)
        ls = losses(conf, disp, clip, y, dt, dm, c, cfg)
        tot += np.array([v.item() for v in ls]) * x.shape[0]
        n += x.shape[0]
        cl.append(torch.sigmoid(clip.float()).cpu())
        fm.append(torch.sigmoid(conf.float()).max(1).values.cpu())
        cs.append(c.cpu())
    c, cp, fp = torch.cat(cs).numpy(), torch.cat(cl).numpy(), torch.cat(fm).numpy()
    return {"loss": tot[0] / n, "loss_conf": tot[1] / n, "loss_clip": tot[2] / n, "loss_disp": tot[3] / n,
            "clip_auc": _safe(roc_auc_score, c, cp), "clip_ap": _safe(average_precision_score, c, cp),
            "fmax_ap": _safe(average_precision_score, c, fp), "n_clips": int(len(c))}


def setup_logger(path):
    log = logging.getLogger("train_v2")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(path)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _worker_init(_):
    torch.set_num_threads(1)


def main():
    p = add_v2_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--name", default=time.strftime("v2_%Y%m%d_%H%M%S"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--init", default=None, help="start from a checkpoint's (EMA) weights, e.g. previous round best.pt")
    p.add_argument("--mined", nargs="*", default=[], help="mined hard-negative spec json files")
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
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

    index, split = load_index(cfg), load_split(cfg)
    mined = []
    for path in args.mined:
        with open(path) as f:
            mined += json.load(f)
    train_ds = TrainClipsV2(cfg, index, mined)
    val_clips = ValClipsV2(cfg, index, "val")
    spot_games = split[tc["spot_subset"]]
    spot_keys = keys_for_games(spot_games)
    gts, ign = gt_dicts(index, spot_keys)
    log.info(f"train/epoch {len(train_ds)} {train_ds.counts} | val clips {len(val_clips)} | "
             f"spot subset {tc['spot_subset']} ({len(spot_games)} games) | device {device}")

    model = build_model_v2(cfg).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        log.info(f"initialised from {args.init}")
    log.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ema = EMA(model, tc["ema_decay"])
    use_fused = (device.type == "cuda")
    opt = torch.optim.AdamW(param_groups(model, tc), fused=use_fused)
    steps = math.ceil(len(train_ds) / tc["batch_size"])
    total, warm = steps * tc["epochs"], max(1, int(tc["warmup"] * steps * tc["epochs"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start, best, bad = 1, float("-inf"), 0
    if args.resume:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_raw"]); ema.model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
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
                            pin_memory=False, drop_last=True, persistent_workers=False,
                            worker_init_fn=_worker_init,
                            generator=torch.Generator().manual_seed(cfg["data"]["seed"] * 1000 + epoch))
        model.train()
        sums, n = np.zeros(4), 0
        n_batches = len(loader)
        for step, (x, y, dt, dm, c) in enumerate(loader):
            x, y, dt, dm, c = (t.to(device, non_blocking=True) for t in (x, y, dt, dm, c))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                conf, disp, clip = model(x)
            ls = losses(conf, disp, clip, y, dt, dm, c, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(ls[0]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                sched.step()
            ema.update(model)
            sums += np.array([v.item() for v in ls]) * x.shape[0]
            n += x.shape[0]
            if (step + 1) % 100 == 0 or (step + 1) == n_batches:
                log.info(f"ep {epoch:03d} [{step+1:03d}/{n_batches}] loss {ls[0].item():.4f}")
        train_s = time.time() - t0

        log.info(f"ep {epoch:03d} evaluating val clips...")
        metrics = {"epoch": epoch, "train/loss": sums[0] / n, "train/loss_conf": sums[1] / n,
                   "train/loss_clip": sums[2] / n, "train/loss_disp": sums[3] / n,
                   "lr_backbone": opt.param_groups[0]["lr"], "time/train_s": train_s}
        metrics.update({f"val/{k}": v for k, v in clip_eval(ema.model, val_clips, cfg, device, 0).items()})
        spotted = epoch % tc["spot_every"] == 0 or epoch == tc["epochs"]
        if spotted:
            log.info(f"ep {epoch:03d} running spotting eval on {len(spot_games)} games...")
            dets = run_spotting(ema.model, cfg, "val", spot_keys, device, tc["spot_stride"], progress=False)
            metrics.update({f"spot/{k}": v for k, v in spot_metrics(dets, gts, ign, cfg).items()})
        metrics["time/epoch_s"] = time.time() - t0

        ckpt = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
        tmp_ckpt = ckpt + ".tmp"
        torch.save({"epoch": epoch, "model": ema.model.state_dict(), "config": cfg, "metrics": metrics}, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt)
        improved = False
        if spotted:
            score = metrics[f"spot/{tc['select_metric']}"]
            score = float("-inf") if score is None or math.isnan(score) else score
            if score > best:
                best, bad, improved = score, 0, True
                shutil.copyfile(ckpt, os.path.join(run_dir, "best.pt"))
            else:
                bad += 1
        metrics.update({"ckpt": ckpt, "best_so_far": best, "is_best": improved})
        last_path = os.path.join(run_dir, "last.pt")
        tmp_last = last_path + ".tmp"
        torch.save({"epoch": epoch, "model": ema.model.state_dict(), "model_raw": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg, "best": best, "bad": bad}, tmp_last)
        os.replace(tmp_last, last_path)
        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                writer.add_scalar(k, v, epoch)
        writer.flush()
        g = lambda k: metrics.get(k, float("nan"))
        msg = (f"ep {epoch:03d} | loss {g('train/loss'):.4f} | val loss {g('val/loss'):.4f} "
               f"clip_auc {g('val/clip_auc'):.4f} clip_ap {g('val/clip_ap'):.4f}")
        if spotted:
            msg += (f" | spot AP5 {g('spot/ap5'):.4f} loose {g('spot/loose_mAP'):.4f} "
                    f"F1@{g('spot/best_thr')} {g('spot/best_f1'):.4f} (P {g('spot/best_precision'):.3f} "
                    f"R {g('spot/best_recall'):.3f})")
        log.info(msg + f" | {g('time/epoch_s'):.0f}s{' *' if improved else ''}")
        if spotted and tc["patience"] and bad >= tc["patience"]:
            log.info(f"early stop: {bad} spotting evaluations without {tc['select_metric']} improvement")
            break
    writer.close()
    log.info(f"done. best spot/{tc['select_metric']} = {best:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
