"""Sliding-window scoring, peak NMS, and spotting metrics (single class)."""
import numpy as np
import torch


def window_starts(N, T, stride):
    if N <= T:
        return [0]
    s = list(range(0, N - T + 1, stride))
    if s[-1] != N - T:
        s.append(N - T)
    return s


@torch.no_grad()
def score_half(model, feats, T, stride, margin, device, batch_windows=512, temperature=1.0):
    """Per-frame card probability for a full half (N, D) -> (N,).

    Overlapping window outputs are averaged. Frames inside the trained jitter range [margin, T-1-margin]
    get weight 1; window edges get weight 1e-3, so they only matter where nothing else covers the frame
    (start/end of half).
    """
    model.eval()
    x = torch.as_tensor(np.asarray(feats, dtype=np.float32))
    N = x.shape[0]
    if N < T:
        x = torch.cat([x, x.new_zeros(T - N, x.shape[1])])
    L = x.shape[0]
    starts = torch.tensor(window_starts(L, T, stride))
    w = torch.full((T,), 1e-3)
    w[margin:T - margin] = 1.0
    acc, cnt = torch.zeros(L), torch.zeros(L)
    x = x.to(device)
    ar = torch.arange(T)
    amp = device.type == "cuda"
    for i in range(0, len(starts), batch_windows):
        idx = starts[i:i + batch_windows, None] + ar[None, :]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            fl, _ = model(x[idx.to(device)])
        p = torch.sigmoid(fl.float() / temperature).cpu()
        acc.index_add_(0, idx.flatten(), (p * w).flatten())
        cnt.index_add_(0, idx.flatten(), w.expand_as(p).flatten())
    return (acc / cnt.clamp_min(1e-12))[:N].numpy()


def nms_peaks(scores, fps, window_s, min_score):
    """Greedy peak picking: take max, suppress +-window_s/2, repeat. Returns [(t_seconds, score)]."""
    s = np.array(scores, dtype=np.float32, copy=True)
    r = int(round(window_s * fps / 2))
    out = []
    while s.size:
        i = int(s.argmax())
        v = float(s[i])
        if v < min_score:
            break
        out.append((i / fps, v))
        s[max(0, i - r):i + r + 1] = -1.0
    return out


def match_detections(dets, gts, ignores, tol):
    """Greedy match by descending score. dets: [(key, t, score)], gts/ignores: {key: array of times}.

    A detection is TP if an unmatched GT lies within tol seconds. Unmatched detections within tol of an
    ignored ('not shown') card are dropped (neither TP nor FP). Returns (scores, tp_flags, n_gt), sorted desc.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    scores, tps = [], []
    for i in order:
        key, t, sc = dets[i]
        g = gts.get(key)
        hit = -1
        if g is not None and len(g):
            d = np.abs(g - t)
            d[used[key]] = np.inf
            j = int(d.argmin())
            if d[j] <= tol:
                hit = j
        if hit >= 0:
            used[key][hit] = True
            scores.append(sc)
            tps.append(1)
            continue
        ig = ignores.get(key)
        if ig is not None and len(ig) and np.min(np.abs(ig - t)) <= tol:
            continue
        scores.append(sc)
        tps.append(0)
    n_gt = int(sum(len(v) for v in gts.values()))
    return np.asarray(scores, dtype=np.float64), np.asarray(tps, dtype=np.int64), n_gt


def average_precision(tps, n_gt):
    """All-point interpolated AP from TP flags sorted by descending score."""
    if n_gt == 0:
        return float("nan")
    if len(tps) == 0:
        return 0.0
    tp = np.cumsum(tps)
    fp = np.cumsum(1 - tps)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def spotting_map(dets, gts, ignores, deltas):
    """SoccerNet-style average-mAP: mean AP over tolerance windows delta (match if |dt| <= delta/2)."""
    per = {float(d): average_precision(match_detections(dets, gts, ignores, d / 2)[1],
                                       int(sum(len(v) for v in gts.values()))) for d in deltas}
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def threshold_grid(dets, gts, ignores, thresholds, tolerances):
    """Precision/recall/F1 for each (threshold, tolerance). Tolerance: |dt| <= tol seconds.

    Greedy matching in descending score order makes thresholding after matching equivalent to
    matching only the detections above the threshold.
    """
    rows = []
    for tol in tolerances:
        scores, tps, n_gt = match_detections(dets, gts, ignores, tol)
        for thr in thresholds:
            keep = scores >= thr
            tp = int(tps[keep].sum())
            fp = int(keep.sum()) - tp
            fn = n_gt - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / n_gt if n_gt else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            rows.append({"thr": round(float(thr), 2), "tol": float(tol), "precision": p, "recall": r,
                         "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return rows


def best_threshold(grid, tol):
    rows = [r for r in grid if r["tol"] == float(tol)]
    return max(rows, key=lambda r: (r["f1"], r["thr"])) if rows else None
