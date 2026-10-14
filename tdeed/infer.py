"""T-DEED inference over cached halves -> SoccerNet-format predictions for the frozen benchmark.

  python -m tdeed.infer --ckpt runs/tdeed_r0/best.pt
  -> runs/tdeed_r0/bench/{raw,calibrated}/{val,test}/<game>/results_spotting.json + calibrator.json
  then:
  venv/bin/python benchmark/benchmark.py --run tdeed_r0 \
      --val-pred runs/tdeed_r0/bench/calibrated/val --test-pred runs/tdeed_r0/bench/calibrated/test

Per half: sliding clip_len windows (50% overlap) -> softmax card prob -> displacement voting (per window)
-> average over overlapping windows -> 1-D NMS (10 s) -> isotonic calibration fitted on val only.

Flip TTA: `--set infer.tta_flip=true` also scores each window mirrored (width flip) and averages the two
voted probabilities. ~2x inference time. Off by default (also for older checkpoints without the key).
Default output dir encodes checkpoint stem and TTA: best.pt -> bench/, avg.pt -> bench_avg/,
best.pt + TTA -> bench_tta/. The printed benchmark run name gets the same suffix.
"""
import argparse
import os

import torch
import yaml
from tqdm import tqdm

from .common import JpegSeq, decode_frames, eval_prefix, get_device, gt_dicts, keys_for_games, load_index
from .model import load_model, vote
from .spotting import nms_peaks, spot_metrics, window_starts, write_bench


@torch.no_grad()
def score_half(model, prefix, cfg, device):
    """Per-frame card probability (N,) for a cached half (<prefix>.data.npy / .off.npy)."""
    d, inf = cfg["data"], cfg["infer"]
    T = int(d["clip_len"])
    stride = max(1, int(round(T * (1 - inf["overlap"]))))
    seq = JpegSeq.open_half(prefix)
    N = len(seq)
    L = max(N, T)
    starts = window_starts(L, T, stride)
    acc = torch.zeros(L, device=device)
    cnt = torch.zeros(L, device=device)
    ar = torch.arange(T, device=device)
    amp = device.type == "cuda"
    tta = bool(inf.get("tta_flip", False))
    model.eval()
    B = int(inf["windows_per_batch"])
    for i in range(0, len(starts), B):
        sts = starts[i:i + B]
        f0 = sts[0]
        frames = decode_frames(seq.raw_range(f0, sts[-1] + T - f0), device)   # (F, 3, H, W) uint8, decoded once
        st = torch.tensor(sts, device=device)
        win = frames[(st - f0)[:, None] + ar[None, :]]                          # (b, T, 3, H, W)
        views = [win, win.flip(-1)] if tta else [win]                           # flip TTA: mirror width
        p = 0.0
        for v in views:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, disp = model(v)
            pv = torch.softmax(logits.float(), dim=-1)[..., 1]
            if inf["vote"]:
                pv = vote(pv, disp.float())
            p = p + pv / len(views)
        g = (st[:, None] + ar[None, :]).flatten()
        acc.index_add_(0, g, p.flatten())
        cnt.index_add_(0, g, torch.ones_like(p).flatten())
    seq.close()
    return (acc / cnt.clamp_min(1e-12))[:N].cpu().numpy()


def detections(prob, cfg, min_score=None):
    inf = cfg["infer"]
    return nms_peaks(prob, cfg["data"]["fps"], inf["nms_window_s"], inf["min_score"] if min_score is None else min_score)


def run_spotting(model, cfg, split, keys, device, progress=True):
    dets = []
    for key in tqdm(keys, desc=f"spot {split}", disable=not progress):
        pre = eval_prefix(cfg, split, key)
        if not os.path.exists(pre + ".off.npy"):
            print(f"[warn] half not cached: {pre}")
            continue
        dets += [(key, t, s) for t, s in detections(score_half(model, pre, cfg, device), cfg)]
    return dets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="default: <ckpt dir>/bench")
    p.add_argument("--device", default="cuda")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="inference overrides on the checkpoint config, e.g. infer.overlap=0.75")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model(args.ckpt, device)
    for kv in args.set:
        k, v = kv.split("=", 1)
        sec, name = k.split(".", 1)
        cfg[sec][name] = yaml.safe_load(v)
    index = load_index(cfg)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = ("" if stem == "best" else f"_{stem}") + ("_tta" if cfg["infer"].get("tta_flip", False) else "")
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"bench{tag}")
    raw = {s: run_spotting(model, cfg, s, keys_for_games(index["splits"][s]), device) for s in ("val", "test")}
    gts, ign = gt_dicts(index, keys_for_games(index["splits"]["val"]))
    print("val (raw scores):", spot_metrics(raw["val"], gts, ign, cfg))
    cal = write_bench(out, index["splits"], raw, gts, ign, float(cfg["eval"]["tolerance_s"]))
    print(f"calibrator fitted on {cal['n']} val detections -> {out}/calibrator.json")
    run = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))) + tag
    print(f"\nnext:\n  venv/bin/python benchmark/benchmark.py --run {run} "
          f"--val-pred {out}/calibrated/val --test-pred {out}/calibrated/test")


if __name__ == "__main__":
    main()
