"""Average the EMA weights of several epoch checkpoints into one checkpoint (removes selection noise).

  venv/bin/python -m tdeed.average --run runs/tdeed_r3 --epochs 3 4 5
  -> runs/tdeed_r3/avg.pt   (same format as best.pt; tdeed.infer writes it to runs/tdeed_r3/bench_avg/)

Why: best.pt is picked by spotting AP on a small val subset, which is noisy (tdeed_r1: 0.793 on the 20-game
subset vs 0.764 on full val). Averaging neighbouring epochs of one run is a cheap, selection-free alternative.

Floating tensors (weights and BatchNorm running stats) are averaged; integer buffers (num_batches_tracked)
come from the last listed epoch. BatchNorm stats are not re-estimated: fine for neighbouring EMA checkpoints
of one run, not for unrelated checkpoints.

Choose between best.pt and avg.pt on VAL numbers only (benchmark val AP / val F1), never on test.
"""
import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run dir containing epoch_XXX.pt")
    p.add_argument("--epochs", nargs="+", type=int, required=True)
    p.add_argument("--out", default=None, help="default: <run>/avg.pt")
    args = p.parse_args()

    paths = [os.path.join(args.run, f"epoch_{e:03d}.pt") for e in args.epochs]
    missing = [x for x in paths if not os.path.exists(x)]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    acc, last = None, None
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        if acc is None:
            acc = {k: v.detach().clone().double() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.double()
                else:
                    acc[k] = v.clone()
        last = ck
    n = len(paths)
    ref = last["model"]
    avg = {k: (v / n).to(ref[k].dtype) if ref[k].is_floating_point() else v for k, v in acc.items()}

    out = args.out or os.path.join(args.run, "avg.pt")
    tmp = out + ".tmp"
    torch.save({"epoch": last["epoch"], "model": avg, "config": last["config"],
                "metrics": {"averaged_epochs": args.epochs, "source": paths}}, tmp)
    os.replace(tmp, out)
    print(f"averaged {n} checkpoints {args.epochs} -> {out}")
    print(f"next: venv/bin/python -m tdeed.infer --ckpt {out}")


if __name__ == "__main__":
    main()
