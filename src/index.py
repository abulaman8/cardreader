"""Build data/index.json and data/splits.json.

- Train games: official SoccerNet 'train' split (300). Training clips are mined ONLY from these.
- Val / test: 20 games each, sampled (seed) from official 'valid' / 'test'. Held out, never mined for training.
- Positives: cards with visibility == 'visible' (Yellow, Red, Yellow->red merged into one class).
- Negatives: centre >= neg_exclusion_s from ANY card (visible or not shown) in the same half.

Usage: python -m src.index [--config ...] [--set key=value ...]
"""
import argparse
import json
import os

import av
import numpy as np
from SoccerNet.utils import getListGames

from .common import HALVES, half_key
from .config import add_config_args, load_config

CARD_LABELS = {"Yellow card", "Red card", "Yellow->red card"}
FOUL_LABELS = {"Foul", "Direct free-kick", "Indirect free-kick"}
STOPPAGE_LABELS = {"Substitution", "Offside", "Kick-off", "Throw-in"}


def video_path(root, game, half):
    return os.path.join(root, game, f"{half}_224p.mkv")


def is_available(root, game):
    return os.path.exists(os.path.join(root, game, "Labels-v2.json")) and all(
        os.path.exists(video_path(root, game, h)) for h in HALVES)


def video_duration(path):
    with av.open(path) as c:
        if c.duration:
            return c.duration / 1e6
        s = c.streams.video[0]
        return float(s.duration * s.time_base)


def load_events(root, game):
    with open(os.path.join(root, game, "Labels-v2.json")) as f:
        ann = json.load(f)["annotations"]
    events = []
    for a in ann:
        half = int(a["gameTime"].split(" - ")[0].strip())
        events.append({"half": half, "t": int(a["position"]) / 1000.0, "label": a["label"],
                       "visibility": a.get("visibility", "visible")})
    return events


def build_halves(root, games):
    """Return halves dict (stored in index) and per-half event arrays (memory only)."""
    halves, aux = {}, {}
    for game in games:
        events = load_events(root, game)
        for h in HALVES:
            key = half_key(game, h)
            ev = [e for e in events if e["half"] == h]
            cards = sorted(({"t": e["t"], "label": e["label"], "visibility": e["visibility"]}
                            for e in ev if e["label"] in CARD_LABELS), key=lambda c: c["t"])
            halves[key] = {"game": game, "half": h, "video": video_path(root, game, h),
                           "duration": video_duration(video_path(root, game, h)), "cards": cards}
            aux[key] = {
                "all_cards": np.array([c["t"] for c in cards], dtype=np.float64),
                "visible": np.array([c["t"] for c in cards if c["visibility"] == "visible"], dtype=np.float64),
                "foul": [e["t"] for e in ev if e["label"] in FOUL_LABELS],
                "stoppage": [e["t"] for e in ev if e["label"] in STOPPAGE_LABELS],
            }
    return halves, aux


def sample_negatives(rng, keys, halves, aux, n, mix, excl, half_win):
    """Sample n negative centres. Hard negatives first, remainder filled with random live play."""

    def valid(key, t):
        if t < half_win or t > halves[key]["duration"] - half_win:
            return False
        cards = aux[key]["all_cards"]
        return cards.size == 0 or np.min(np.abs(cards - t)) >= excl

    out = []

    def draw_events(kind, lo, hi, count):
        cands = [(k, t) for k in keys for t in aux[k][kind] if valid(k, t)]
        got, tries = 0, 0
        while cands and got < count and tries < 20 * count:
            tries += 1
            k, t = cands[int(rng.integers(len(cands)))]
            c = t + rng.uniform(lo, hi)
            if valid(k, c):
                out.append({"key": k, "t": round(float(c), 2), "kind": kind})
                got += 1
        return got

    # foul: centre 0-10 s after the foul, i.e. where a card would have been shown
    got_f = draw_events("foul", 0.0, 10.0, int(round(n * mix.get("foul", 0))))
    got_s = draw_events("stoppage", -2.0, 2.0, int(round(n * mix.get("stoppage", 0))))

    durations = np.array([halves[k]["duration"] for k in keys])
    probs = durations / durations.sum()
    need = n - got_f - got_s
    while need > 0:
        k = keys[int(rng.choice(len(keys), p=probs))]
        t = rng.uniform(0, halves[k]["duration"])
        if valid(k, t):
            out.append({"key": k, "t": round(float(t), 2), "kind": "random"})
            need -= 1
    return out


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    d, root = cfg["data"], cfg["paths"]["data_root"]
    fps, L, sigma = d["fps"], d["cache_len"], cfg["targets"]["sigma"]
    cache_s, half_win = L / fps, L / fps / 2
    rng = np.random.default_rng(d["seed"])

    avail = {s: sorted(g for g in getListGames(s) if is_available(root, g)) for s in ("train", "valid", "test")}
    for s, gs in avail.items():
        print(f"{s}: {len(gs)} games available locally")
    train_games = avail["train"]
    val_games = sorted(rng.choice(avail["valid"], d["n_val_games"], replace=False).tolist())
    test_games = sorted(rng.choice(avail["test"], d["n_test_games"], replace=False).tolist())
    assert not (set(train_games) & set(val_games)), "train/val overlap"
    assert not (set(train_games) & set(test_games)), "train/test overlap"
    assert not (set(val_games) & set(test_games)), "val/test overlap"
    splits = {"train": train_games, "val": val_games, "test": test_games}

    print("probing video durations ...")
    halves, aux = build_halves(root, train_games + val_games + test_games)
    keys = {s: [half_key(g, h) for g in gs for h in HALVES] for s, gs in splits.items()}

    # ---- train positives: one cached 9 s window per visible card (train games only) ----
    train_pos = []
    for key in keys["train"]:
        dur, vis = halves[key]["duration"], aux[key]["visible"]
        for t in vis:
            start = round(float(np.clip(t - half_win, 0.0, max(dur - cache_s, 0.0))), 2)
            # include cards slightly outside the window so Gaussian tails are correct
            rel = [int(round((c - start) * fps)) for c in vis if -3 * sigma <= (c - start) * fps < L + 3 * sigma]
            train_pos.append({"id": f"pos_{len(train_pos):05d}", "key": key, "start": start,
                              "anchor": int(round((t - start) * fps)), "cards": rel})

    # ---- train negatives (train games only) ----
    negs = sample_negatives(rng, keys["train"], halves, aux, d["neg_pool"], d["neg_mix"],
                            d["neg_exclusion_s"], half_win)
    train_neg = [{"id": f"neg_{i:05d}", "key": n["key"], "t": n["t"], "kind": n["kind"],
                  "start": round(n["t"] - half_win, 2)} for i, n in enumerate(negs)]

    # ---- held-out val/test clip specs (sliced from full-half features at eval time) ----
    eval_specs = {}
    for split in ("val", "test"):
        pos = [{"key": k, "t": float(t), "u": float(rng.random())} for k in keys[split] for t in aux[k]["visible"]]
        neg = sample_negatives(rng, keys[split], halves, aux, d["eval_neg_per_pos"] * len(pos), d["neg_mix"],
                               d["neg_exclusion_s"], d["window"] / fps / 2)
        eval_specs[split] = {"pos": pos, "neg": neg}

    index = {"splits": splits, "halves": halves, "train_pos": train_pos, "train_neg": train_neg,
             "eval": eval_specs, "fps": fps, "cache_len": L}
    for p in (cfg["paths"]["index"], cfg["paths"]["splits"]):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(cfg["paths"]["index"], "w") as f:
        json.dump(index, f)
    with open(cfg["paths"]["splits"], "w") as f:
        json.dump(splits, f, indent=1)

    kinds = {}
    for n in train_neg:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    print(f"games: train {len(train_games)} | val {len(val_games)} | test {len(test_games)}")
    print(f"train positives: {len(train_pos)} | train negatives: {len(train_neg)} {kinds}")
    for s in ("val", "test"):
        print(f"{s} clips: pos {len(eval_specs[s]['pos'])} | neg {len(eval_specs[s]['neg'])}")
    print(f"wrote {cfg['paths']['index']} and {cfg['paths']['splits']}")


if __name__ == "__main__":
    main()
