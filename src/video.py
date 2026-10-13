"""Video decoding with PyAV: read n frames at a target fps starting at start_s."""
import av
import numpy as np


def read_frames(path, start_s, n, fps):
    """Return uint8 array (n, H, W, 3). Frame k is the source frame nearest to start_s + k/fps.

    Times are relative to the stream start. Pads with the last frame if the video ends early.
    """
    targets = start_s + np.arange(n) / fps
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        t0 = stream.start_time or 0
        src_fps = float(stream.average_rate) if stream.average_rate else 25.0
        half_frame = 0.5 / src_fps
        if start_s > 1.0:
            # seek a bit early; seek lands on the preceding keyframe, decode forward from there
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
