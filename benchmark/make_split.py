"""Create benchmark/split.json ONCE. Refuses to overwrite.

400 train / 50 val / 50 test games.
- val  = v1's 20 val games  + 30 more from official 'valid' (seed 0)
- test = v1's 20 test games + 30 more from official 'test'  (seed 0)
- train = official 'train' (300) + remaining 50 'valid' + remaining 50 'test'

Usage (from repo root): venv/bin/python benchmark/make_split.py
"""
import datetime
import json
import os
import sys

import numpy as np
from SoccerNet.utils import getListGames

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "split.json")
DATA_ROOT = "soccernet_data"
V1_SPLITS = "data/splits.json"
N_VAL, N_TEST, SEED = 50, 50, 0


def available(game):
    d = os.path.join(DATA_ROOT, game)
    return all(os.path.exists(os.path.join(d, f)) for f in ("Labels-v2.json", "1_224p.mkv", "2_224p.mkv"))


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} already exists. The benchmark split is frozen; delete it manually only if you intend "
                 f"to invalidate every previous benchmark result.")
    with open(V1_SPLITS) as f:
        v1 = json.load(f)
    rng = np.random.default_rng(SEED)
    official = {s: sorted(g for g in getListGames(s) if available(g)) for s in ("train", "valid", "test")}

    def extend(base, pool, n):
        rest = sorted(set(pool) - set(base))
        extra = rng.choice(rest, n - len(base), replace=False).tolist()
        return sorted(base + extra)

    val = extend(list(v1["val"]), official["valid"], N_VAL)
    test = extend(list(v1["test"]), official["test"], N_TEST)
    train = sorted(official["train"] + sorted(set(official["valid"]) - set(val)) + sorted(set(official["test"]) - set(test)))
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert set(v1["val"]) <= set(val) and set(v1["test"]) <= set(test)
    split = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
             "train": train, "val": val, "test": test, "val_v1": sorted(v1["val"]), "test_v1": sorted(v1["test"])}
    with open(OUT, "w") as f:
        json.dump(split, f, indent=1)
    print(f"wrote {OUT}: train {len(train)} | val {len(val)} | test {len(test)}")


if __name__ == "__main__":
    main()
