"""Sanity baseline: logistic regression on mean-pooled clip features (no temporal modelling).

The transformer must beat this on val/test clip AUC/AP, otherwise temporal modelling adds nothing.
  python -m src.baseline
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .common import load_index
from .config import add_config_args, load_config
from .dataset import EvalClips, TrainClips


def pooled(ds, workers):
    X, Y = [], []
    for x, _, c in DataLoader(ds, batch_size=256, shuffle=False, num_workers=workers):
        X.append(x.mean(1).numpy())
        Y.append(c.numpy())
    return np.concatenate(X), np.concatenate(Y)


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    index = load_index(cfg)
    w = cfg["train"]["num_workers"]

    train = TrainClips(cfg, index)
    Xtr, ytr = pooled(train, w)
    print(f"train: {len(ytr)} clips ({int(ytr.sum())} pos)")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(Xtr, ytr)

    results = {}
    for split in ("val", "test"):
        ds = EvalClips(cfg, index, split)
        if not len(ds):
            print(f"[warn] {split}: no clips (extract halves first)")
            continue
        X, y = pooled(ds, 0)
        s = clf.predict_proba(X)[:, 1]
        results[split] = {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
                          "n": int(len(y)), "n_pos": int(y.sum())}
        print(f"{split}: AUC {results[split]['auc']:.4f} AP {results[split]['ap']:.4f} "
              f"(n {len(y)}, pos {int(y.sum())})")
    out = args.out or os.path.join(cfg["paths"]["runs"], "baseline", "metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
