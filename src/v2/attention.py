"""Save spatial-attention overlays for val cards, to check the model looks at the referee / card.

  python -m src.v2.attention --ckpt runs/v2_r0/best.pt --n 30
  -> runs/v2_r0/attn/<i>_<game>_h<half>_<t>.png  (frame at the card + one overlay per attention head)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import get_device, half_prefix, load_index
from .frames import JpegSeq
from .model import load_model_v2


def overlay(frame, att):
    """frame uint8 (3,H,W), att (h,w) -> RGB uint8 (H,W,3) with red heat overlay."""
    H, W = frame.shape[1:]
    a = F.interpolate(att[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    img = frame.permute(1, 2, 0).float()
    heat = torch.zeros_like(img)
    heat[..., 0] = 255 * a
    return (0.55 * img + 0.45 * heat).clamp(0, 255).byte().numpy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, _ = load_model_v2(args.ckpt, device)
    index = load_index(cfg)
    d = cfg["data"]
    T, m, fps = d["window"], d["margin"], d["fps"]
    out = os.path.join(os.path.dirname(args.ckpt), "attn")
    os.makedirs(out, exist_ok=True)
    for i, c in enumerate(index["eval"]["val"]["pos"][:args.n]):
        pre = half_prefix(cfg, "val", c["key"])
        if not os.path.exists(pre + ".off.npy"):
            continue
        seq = JpegSeq.open_half(pre)
        cf = round(c["t"] * fps)
        s = int(np.clip(cf - T // 2, 0, max(len(seq) - T, 0)))
        x = seq.decode(s, T)
        conf, _, _, att = model(x[None].to(device), return_attn=True)
        k = int(np.clip(cf - s, 0, T - 1))
        frame, a = x[k], att[0, k].float().cpu()                            # (heads, h, w)
        tiles = [frame.permute(1, 2, 0).numpy()] + [overlay(frame, a[h]) for h in range(a.shape[0])]
        game, half = c["key"].rsplit("|", 1)
        name = f"{i:03d}_{game.split('/')[-1][:40].replace(' ', '_')}_h{half}_{c['t']:.0f}s.png"
        Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(out, name))
        print(f"{name}  conf@card {torch.sigmoid(conf[0, k]).item():.3f}")
    print(f"overlays in {out}")


if __name__ == "__main__":
    main()
