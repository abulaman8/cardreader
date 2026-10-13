"""Extract frozen DINOv2 features (fp16) to disk. Frames are never written to disk.

  python -m src.extract clips                 # train positives + negatives -> features/train/{pos,neg}/<id>.npy  (2, cache_len, 768): [orig, h-flip]
  python -m src.extract halves --split val test   # full held-out halves   -> features/eval/<split>/<game>/half{1,2}.npy  (N, 768), frame k at k/fps s

Resumable: existing output files are skipped; writes are atomic.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import get_device, half_feat_path, load_index, split_keys, train_feat_path
from .config import add_config_args, load_config
from .encoder import DinoEncoder
from .video import read_frames


def save_atomic(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


class FrameJobs(Dataset):
    """Each job: (tag, video, start_s, n). Decoded in DataLoader workers."""

    def __init__(self, jobs, fps):
        self.jobs, self.fps = jobs, fps

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        tag, video, start_s, n = self.jobs[i]
        try:
            return tag, read_frames(video, start_s, n, self.fps)
        except Exception as e:  # keep going; report in main process
            print(f"[warn] decode failed {video} @ {start_s:.2f}s: {e}")
            return tag, None


def make_loader(jobs, fps, workers):
    return DataLoader(FrameJobs(jobs, fps), batch_size=None, shuffle=False, num_workers=workers,
                      prefetch_factor=2 if workers > 0 else None)


def extract_clips(cfg, index, enc, limit=None):
    fps, L = cfg["data"]["fps"], cfg["data"]["cache_len"]
    jobs = []
    for kind, clips in (("pos", index["train_pos"]), ("neg", index["train_neg"])):
        for c in clips:
            out = train_feat_path(cfg, kind, c["id"])
            if not os.path.exists(out):
                jobs.append((out, index["halves"][c["key"]]["video"], c["start"], L))
    if limit:
        jobs = jobs[:limit]
    print(f"clips to extract: {len(jobs)}")
    failed = 0
    for out, frames in tqdm(make_loader(jobs, fps, cfg["encoder"]["num_workers"]), total=len(jobs)):
        if frames is None:
            failed += 1
            continue
        feats = torch.stack([enc.encode(frames), enc.encode(frames, flip=True)])
        save_atomic(out, feats.numpy().astype(np.float16))
    print(f"done, failed: {failed}")


def extract_halves(cfg, index, enc, split, limit=None):
    fps, chunk = cfg["data"]["fps"], cfg["encoder"]["chunk_frames"]
    keys = split_keys(index, split)
    if limit:
        keys = keys[:limit]
    jobs, meta = [], {}
    for key in keys:
        out = half_feat_path(cfg, split, key)
        if os.path.exists(out):
            continue
        h = index["halves"][key]
        N = int(h["duration"] * fps)
        starts = list(range(0, N, chunk))
        meta[key] = (out, N, len(starts))
        jobs += [((key, s), h["video"], s / fps, min(chunk, N - s)) for s in starts]
    print(f"[{split}] halves to extract: {len(meta)} ({len(jobs)} chunks)")
    buffers, remaining = {}, {}
    for (key, s), frames in tqdm(make_loader(jobs, fps, cfg["encoder"]["num_workers"]), total=len(jobs)):
        out, N, nchunks = meta[key]
        if key not in buffers:
            buffers[key] = np.zeros((N, enc.dim), dtype=np.float16)
            remaining[key] = nchunks
        if frames is not None:
            buffers[key][s:s + len(frames)] = enc.encode(frames).numpy()
        else:
            print(f"[warn] {key}: chunk at frame {s} left as zeros")
        remaining[key] -= 1
        if remaining[key] == 0:
            save_atomic(out, buffers.pop(key))
            del remaining[key]


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("mode", choices=["clips", "halves"])
    p.add_argument("--split", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="only process the first N jobs (clips) / halves")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    index = load_index(cfg)
    enc = DinoEncoder(cfg["encoder"]["name"], get_device(args.device), cfg["encoder"]["batch_frames"])
    assert enc.dim == cfg["encoder"]["dim"], f"encoder dim {enc.dim} != config {cfg['encoder']['dim']}"
    if args.mode == "clips":
        extract_clips(cfg, index, enc, args.limit)
    else:
        for split in args.split:
            extract_halves(cfg, index, enc, split, args.limit)


if __name__ == "__main__":
    main()
