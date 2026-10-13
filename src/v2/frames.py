"""JPEG frame cache at 5 fps (the end-to-end model needs pixels).

  python -m src.v2.frames clips                    # train pos + neg pool -> frames/train/{pos,neg}/<id>.npz
  python -m src.v2.frames clips --kind mined --mined data/v2/mined_round1.json
  python -m src.v2.frames halves --split val test  # benchmark halves -> frames/eval/<split>/<game>/half{1,2}.{data,off}.npy

Clip file (.npz): data = concatenated JPEG bytes (uint8), off = int64 offsets (n+1).
Half files: <prefix>.data.npy (memory-mapped) + <prefix>.off.npy. Resumable, atomic writes.
"""
import argparse
import json
import multiprocessing
import os

import av
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg
from tqdm import tqdm

from ..video import read_frames
from .common import add_v2_config_args, clip_path, half_prefix, keys_for_games, load_config, load_index


class JpegSeq:
    """Random access to a JPEG frame sequence. decode() returns uint8 (n, 3, H, W)."""

    def __init__(self, data, off):
        self.data, self.off = data, off

    @classmethod
    def open_half(cls, prefix):
        return cls(np.load(prefix + ".data.npy", mmap_mode="r"), np.load(prefix + ".off.npy"))

    @classmethod
    def open_clip(cls, path):
        with np.load(path) as z:
            return cls(z["data"], z["off"])

    def __len__(self):
        return len(self.off) - 1

    def decode(self, start, n):
        idx = np.clip(np.arange(start, start + n), 0, len(self) - 1)
        out, cache = [], {}
        for i in idx:
            if i not in cache:
                buf = torch.from_numpy(np.array(self.data[self.off[i]:self.off[i + 1]]))
                cache[i] = decode_jpeg(buf)
            out.append(cache[i])
        return torch.stack(out)

    def close(self):
        if hasattr(self.data, "_mmap") and self.data._mmap is not None:
            try:
                self.data._mmap.close()
            except Exception:
                pass


def half_len(prefix):
    return len(np.load(prefix + ".off.npy")) - 1


def _encode(frames, h, w, quality):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()          # (n, 3, H, W) uint8
    if tuple(x.shape[-2:]) != (h, w):
        x = F.interpolate(x.float(), size=(h, w), mode="bilinear", antialias=True, align_corners=False)
        x = x.round().clamp(0, 255).to(torch.uint8)
    enc = [encode_jpeg(f, quality=quality).numpy() for f in x]
    off = np.zeros(len(enc) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(e) for e in enc])
    return np.concatenate(enc), off


def _atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _worker_init():
    torch.set_num_threads(1)


def _clip_worker(job):
    out, video, start, L, fps, h, w, q = job
    if os.path.exists(out):
        return True
    try:
        frames = read_frames(video, start, L, fps)
        data, off = _encode(frames, h, w, q)
        _atomic_npz(out, data=data, off=off)
        return True
    except Exception as e:
        print(f"[warn] failed {video} @ {start:.2f}s: {e}")
        return False


def extract_clips(cfg, index, specs, kind, workers):
    L = cfg["data"]["cache_len"]
    d = cfg["data"]
    jobs = [(clip_path(cfg, kind, c["id"]), index["halves"][c["key"]]["video"], c["start"],
             L, d["fps"], d["frame_h"], d["frame_w"], d["jpeg_quality"])
            for c in specs if not os.path.exists(clip_path(cfg, kind, c["id"]))]
    print(f"[{kind}] clips to cache: {len(jobs)}")
    if not jobs:
        return
    failed = 0
    with multiprocessing.Pool(workers or 1, initializer=_worker_init) as pool:
        for ok in tqdm(pool.imap_unordered(_clip_worker, jobs, chunksize=16), total=len(jobs)):
            if not ok:
                failed += 1
    print(f"[{kind}] done, failed {failed}")


def _half_worker(job):
    prefix, video, fps, h, w, q = job
    if os.path.exists(prefix + ".off.npy"):
        return True
    try:
        with av.open(video) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            t0 = stream.start_time or 0
            src_fps = float(stream.average_rate) if stream.average_rate else 25.0
            half_frame = 0.5 / src_fps
            next_target = 0.0
            enc_list = []
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = (frame.pts - t0) * tb
                if t < next_target - half_frame:
                    continue
                img = frame.to_ndarray(format="rgb24")
                while t >= next_target - half_frame:
                    tensor = torch.from_numpy(img).permute(2, 0, 1)
                    if tuple(tensor.shape[-2:]) != (h, w):
                        tensor = F.interpolate(tensor.unsqueeze(0).float(), size=(h, w), mode="bilinear",
                                               antialias=True, align_corners=False).squeeze(0).round().clamp(0, 255).to(torch.uint8)
                    enc_list.append(encode_jpeg(tensor, quality=q).numpy())
                    next_target += 1.0 / fps
        if not enc_list:
            raise RuntimeError(f"no frames decoded from {video}")
        off = np.zeros(len(enc_list) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(e) for e in enc_list])
        data = np.concatenate(enc_list)
        _atomic_npy(prefix + ".data.npy", data)
        _atomic_npy(prefix + ".off.npy", off)
        return True
    except Exception as e:
        print(f"[warn] failed half {video}: {e}")
        return False


def extract_halves(cfg, index, keys, prefix_of, workers):
    d = cfg["data"]
    jobs = []
    for key in keys:
        prefix = prefix_of(key)
        if os.path.exists(prefix + ".off.npy"):
            continue
        jobs.append((prefix, index["halves"][key]["video"], d["fps"], d["frame_h"], d["frame_w"], d["jpeg_quality"]))
    if jobs:
        print(f"halves to cache: {len(jobs)}")
    if not jobs:
        return [prefix_of(k) for k in keys]
    failed = 0
    with multiprocessing.Pool(min(workers or 1, len(jobs)), initializer=_worker_init) as pool:
        for ok in tqdm(pool.imap_unordered(_half_worker, jobs, chunksize=1), total=len(jobs)):
            if not ok:
                failed += 1
    if failed > 0:
        print(f"[warn] {failed} halves failed")
    return [prefix_of(k) for k in keys]


def main():
    p = add_v2_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("mode", choices=["clips", "halves"])
    p.add_argument("--kind", default="posneg", choices=["posneg", "mined"])
    p.add_argument("--mined", nargs="*", default=[], help="mined spec json files (with --kind mined)")
    p.add_argument("--split", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    index = load_index(cfg)
    if args.mode == "clips":
        if args.kind == "posneg":
            extract_clips(cfg, index, index["train_pos"], "pos", args.workers)
            extract_clips(cfg, index, index["train_neg"], "neg", args.workers)
        else:
            for path in args.mined:
                with open(path) as f:
                    extract_clips(cfg, index, json.load(f), "mined", args.workers)
    else:
        for split in args.split:
            keys = keys_for_games(index["splits"][split])
            extract_halves(cfg, index, keys, lambda k, s=split: half_prefix(cfg, s, k), args.workers)


if __name__ == "__main__":
    main()
