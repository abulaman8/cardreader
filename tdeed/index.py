"""Build data/tdeed/index.json: halves (cards, fouls, stoppages), long train windows, val clip specs.

  python -m tdeed.index

- Train windows come ONLY from the 400 benchmark train games.
- Positive window: window_s (40 s) centred on each visible card (clipped to the half). The model clip
  (clip_len frames) is cropped at train time so the card can sit anywhere inside it.
- Negative window: centre drawn from {random live play, 0-10 s after a foul, +-2 s around a stoppage};
  the WHOLE window must be >= neg_exclusion_s from any card (visible or not shown).
- Val clip specs (clip-level loss during training) are sliced from the existing val half cache.
"""
import argparse
import json
import os

import av
import numpy as np

from .common import (CARD_LABELS, FOUL_LABELS, HALVES, STOPPAGE_LABELS, add_config_args, half_key, keys_for_games,
                     load_config, load_split)


def video_path(root, game, half):
    return os.path.join(root, game, f"{half}_224p.mkv")


def video_duration(path):
    with av.open(path) as c:
        if c.duration:
            return c.duration / 1e6
        s = c.streams.video[0]
        return float(s.duration * s.time_base)


def load_events(root, game):
    with open(os.path.join(root, game, "Labels-v2.json")) as f:
        ann = json.load(f)["annotations"]
    return [{"half": int(a["gameTime"].split(" - ")[0].strip()), "t": int(a["position"]) / 1000.0,
             "label": a["label"], "visibility": a.get("visibility", "visible")} for a in ann]


def build_halves(root, games):
    halves = {}
    for game in games:
        events = load_events(root, game)
        for h in HALVES:
            ev = [e for e in events if e["half"] == h]
            cards = sorted(({"t": e["t"], "label": e["label"], "visibility": e["visibility"]}
                            for e in ev if e["label"] in CARD_LABELS), key=lambda c: c["t"])
            halves[half_key(game, h)] = {
                "game": game, "half": h, "video": video_path(root, game, h),
                "duration": video_duration(video_path(root, game, h)), "cards": cards,
                "foul": sorted(e["t"] for e in ev if e["label"] in FOUL_LABELS),
                "stoppage": sorted(e["t"] for e in ev if e["label"] in STOPPAGE_LABELS)}
    return halves


def card_array(hv, visible=None):
    """visible=None -> all cards; True -> visible; False -> not shown."""
    return np.array([c["t"] for c in hv["cards"]
                     if visible is None or (c["visibility"] == "visible") == visible], dtype=np.float64)


def window_far_from_cards(hv, start, length, excl):
    """True if every card is >= excl seconds from the interval [start, start + length]."""
    c = card_array(hv)
    if c.size == 0:
        return True
    d = np.maximum(np.maximum(start - c, c - (start + length)), 0.0)
    return float(d.min()) >= excl


def rel_frames(times, start, n, fps, margin):
    """Frame indices (relative to start) of event times within [-margin, n + margin) frames."""
    out = []
    for t in times:
        f = (t - start) * fps
        if -margin <= f < n + margin:
            out.append(int(round(f)))
    return out


def sample_neg_windows(rng, keys, halves, n, mix, length, excl):
    """n negative windows [(key, start_s, kind)]: foul / stoppage first, remainder random live play."""
    out = []

    def ok(k, centre):
        s = centre - length / 2
        return 0.0 <= s <= halves[k]["duration"] - length and window_far_from_cards(halves[k], s, length, excl)

    def draw(kind, lo, hi, count):
        cands = [(k, t) for k in keys for t in halves[k][kind]]
        got, tries = 0, 0
        while cands and got < count and tries < 50 * max(count, 1):
            tries += 1
            k, t = cands[int(rng.integers(len(cands)))]
            c = t + rng.uniform(lo, hi)
            if ok(k, c):
                out.append((k, round(float(c - length / 2), 2), kind))
                got += 1
        return got

    got_f = draw("foul", 0.0, 10.0, int(round(n * mix.get("foul", 0))))
    got_s = draw("stoppage", -2.0, 2.0, int(round(n * mix.get("stoppage", 0))))
    dur = np.array([halves[k]["duration"] for k in keys])
    p = dur / dur.sum()
    need, tries = n - got_f - got_s, 0
    while need > 0 and tries < 200 * n:
        tries += 1
        k = keys[int(rng.choice(len(keys), p=p))]
        c = rng.uniform(length / 2, max(length / 2, halves[k]["duration"] - length / 2))
        if ok(k, c):
            out.append((k, round(float(c - length / 2), 2), "random"))
            need -= 1
    if need > 0:
        print(f"[warn] only {n - need}/{n} negative windows found")
    return out


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__,
                                                   formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    cfg = load_config(args.config, args.set)
    d, root = cfg["data"], cfg["paths"]["data_root"]
    fps, W, T = d["fps"], float(d["window_s"]), int(d["clip_len"])
    n_win = int(round(W * fps))
    R = int(cfg["targets"]["radius"])
    ign = int(round(d["ignore_radius_s"] * fps))
    rng = np.random.default_rng(d["seed"])

    split = load_split(cfg)
    train, val, test = split["train"], split["val"], split["test"]
    assert not (set(train) & (set(val) | set(test))), "train overlaps benchmark games"
    print(f"games: train {len(train)} | val {len(val)} | test {len(test)}; probing durations ...")
    halves = build_halves(root, train + val + test)
    tkeys, vkeys = keys_for_games(train), keys_for_games(val)

    # ---- positive windows (train only)
    train_pos = []
    for key in tkeys:
        hv = halves[key]
        vis, hid = card_array(hv, True), card_array(hv, False)
        for t in vis:
            start = round(float(np.clip(t - W / 2, 0.0, max(hv["duration"] - W, 0.0))), 2)
            train_pos.append({"id": f"pos_{len(train_pos):05d}", "key": key, "start": start, "n": n_win,
                              "anchor": int(round((t - start) * fps)),
                              "cards": rel_frames(vis, start, n_win, fps, R),
                              "ignore": rel_frames(hid, start, n_win, fps, ign)})

    # ---- negative windows (train only)
    negs = sample_neg_windows(rng, tkeys, halves, int(d["neg_pool"]), d["neg_mix"], W, d["neg_exclusion_s"])
    train_neg = [{"id": f"neg_{i:05d}", "key": k, "start": s, "n": n_win, "kind": kind, "anchor": None,
                  "cards": [], "ignore": []} for i, (k, s, kind) in enumerate(negs)]

    # ---- val clip specs (frame starts in the val half cache, frame k <-> k / fps seconds)
    val_clips = []
    for key in vkeys:
        hv = halves[key]
        N = int(hv["duration"] * fps)
        for t in card_array(hv, True):
            off = int(rng.integers(0, T))
            s = int(np.clip(int(round(t * fps)) - off, 0, max(N - T, 0)))
            val_clips.append({"key": key, "start": s, "pos": True})
    n_vpos = len(val_clips)
    for k, s, kind in sample_neg_windows(rng, vkeys, halves, int(round(d["val_neg_per_pos"] * n_vpos)),
                                         d["neg_mix"], T / fps, d["neg_exclusion_s"]):
        val_clips.append({"key": k, "start": int(round(s * fps)), "pos": False, "kind": kind})

    index = {"version": "tdeed-1", "fps": fps, "window_s": W, "clip_len": T,
             "splits": {k: split[k] for k in ("train", "val", "test", "val_v1", "test_v1") if k in split},
             "halves": halves, "train_pos": train_pos, "train_neg": train_neg, "val_clips": val_clips}
    os.makedirs(os.path.dirname(cfg["paths"]["index"]) or ".", exist_ok=True)
    tmp = cfg["paths"]["index"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, cfg["paths"]["index"])

    kinds = {}
    for n in train_neg:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    gb = (len(train_pos) + len(train_neg)) * n_win * 22e3 / 1e9
    print(f"train windows: pos {len(train_pos)} | neg {len(train_neg)} {kinds} | ~{gb:.0f} GB to extract")
    print(f"val clips: pos {n_vpos} | neg {len(val_clips) - n_vpos}")
    print(f"wrote {cfg['paths']['index']}")


if __name__ == "__main__":
    main()
