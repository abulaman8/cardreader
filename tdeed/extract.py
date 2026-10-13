"""Frame extraction for the T-DEED replica (5 fps JPEG q90, 224x398, same format as the existing cache).

  python -m tdeed.extract windows                         # train pos + neg windows -> frames_tdeed/train/{pos,neg}/<id>.npz
  python -m tdeed.extract windows --kind mined --specs data/tdeed/mined_round1.json
  python -m tdeed.extract eval-halves                     # only fills MISSING val/test halves (never overwrites)

Window file (.npz): data = concatenated JPEG bytes (uint8), off = int64 offsets (n+1).
Writes are atomic and resumable (existing files are skipped).
"""
import argparse
import json
import multiprocessing
import os

import av
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import encode_jpeg
from tqdm import tqdm

from .common import add_config_args, eval_prefix, keys_for_games, load_config, load_index, window_path


def read_frames(path, start_s, n, fps):
    """uint8 (n, H, W, 3). Frame k is the source frame nearest to start_s + k / fps. Pads with the last frame."""
    targets = start_s + np.arange(n) / fps
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        t0 = stream.start_time or 0
        half_frame = 0.5 / (float(stream.average_rate) if stream.average_rate else 25.0)
        if start_s > 1.0:
            container.seek(int((start_s - 1.0) / tb) + t0, stream=stream, backward=True, any_frame=False)
        k = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = (frame.pts - t0) * tb
            if t < targets[k] - half_frame:
                continue
            img = frame.to_ndarray(format="rgb24")
            while k < n and t >= targets[k] - half_frame:
                frames.append(img)
                k += 1
            if k >= n:
                break
    if not frames:
        raise RuntimeError(f"no frames decoded from {path} at {start_s:.2f}s")
    while len(frames) < n:
        frames.append(frames[-1])
    return np.stack(frames)


def _resize(x, h, w):
    """uint8 (n, 3, H, W) -> (n, 3, h, w)."""
    if tuple(x.shape[-2:]) == (h, w):
        return x
    x = F.interpolate(x.float(), size=(h, w), mode="bilinear", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)


def encode_frames(frames, h, w, quality):
    x = _resize(torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(), h, w)
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


def _window_worker(job):
    out, video, start, n, fps, h, w, q = job
    if os.path.exists(out):
        return True
    try:
        data, off = encode_frames(read_frames(video, start, n, fps), h, w, q)
        _atomic_npz(out, data=data, off=off)
        return True
    except Exception as e:
        print(f"[warn] failed {video} @ {start:.2f}s: {e}")
        return False


def extract_windows(cfg, index, specs, kind, workers):
    d = cfg["data"]
    jobs = [(window_path(cfg, kind, s["id"]), index["halves"][s["key"]]["video"], s["start"], s["n"],
             d["fps"], d["frame_h"], d["frame_w"], d["jpeg_quality"])
            for s in specs if not os.path.exists(window_path(cfg, kind, s["id"]))]
    print(f"[{kind}] windows to cache: {len(jobs)} / {len(specs)}")
    if not jobs:
        return
    failed = 0
    with multiprocessing.Pool(max(1, workers), initializer=_worker_init) as pool:
        for ok in tqdm(pool.imap_unordered(_window_worker, jobs, chunksize=4), total=len(jobs)):
            failed += not ok
    print(f"[{kind}] done, failed {failed}")


def cache_half(video, prefix, fps, h, w, q):
    """Decode a full half at `fps` into <prefix>.data.npy + <prefix>.off.npy (same layout as frames/eval)."""
    try:
        with av.open(video) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            t0 = stream.start_time or 0
            half_frame = 0.5 / (float(stream.average_rate) if stream.average_rate else 25.0)
            nxt, enc = 0.0, []
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = (frame.pts - t0) * tb
                if t < nxt - half_frame:
                    continue
                x = _resize(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)[None], h, w)[0]
                while t >= nxt - half_frame:
                    enc.append(encode_jpeg(x, quality=q).numpy())
                    nxt += 1.0 / fps
        if not enc:
            raise RuntimeError(f"no frames decoded from {video}")
        off = np.zeros(len(enc) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(e) for e in enc])
        _atomic_npy(prefix + ".data.npy", np.concatenate(enc))
        _atomic_npy(prefix + ".off.npy", off)
        return True
    except Exception as e:
        print(f"[warn] failed half {video}: {e}")
        return False


def _half_job(job):
    return cache_half(*job)


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("mode", choices=["windows", "eval-halves"])
    p.add_argument("--kind", default="posneg", choices=["posneg", "mined"])
    p.add_argument("--specs", nargs="*", default=[], help="mined window spec json files (with --kind mined)")
    p.add_argument("--split", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    index = load_index(cfg)
    d = cfg["data"]
    if args.mode == "windows":
        if args.kind == "posneg":
            extract_windows(cfg, index, index["train_pos"], "pos", args.workers)
            extract_windows(cfg, index, index["train_neg"], "neg", args.workers)
        else:
            for path in args.specs:
                with open(path) as f:
                    extract_windows(cfg, index, json.load(f), "mined", args.workers)
    else:
        jobs = []
        for split in args.split:
            for key in keys_for_games(index["splits"][split]):
                pre = eval_prefix(cfg, split, key)
                if not os.path.exists(pre + ".off.npy"):
                    jobs.append((index["halves"][key]["video"], pre, d["fps"], d["frame_h"], d["frame_w"],
                                 d["jpeg_quality"]))
        print(f"missing eval halves: {len(jobs)}")
        if jobs:
            with multiprocessing.Pool(min(max(1, args.workers), len(jobs)), initializer=_worker_init) as pool:
                fails = sum(not ok for ok in tqdm(pool.imap_unordered(_half_job, jobs), total=len(jobs)))
            print(f"done, failed {fails}")


if __name__ == "__main__":
    main()
