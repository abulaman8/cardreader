"""Frozen DINOv2 frame encoder. Output per frame: CLS token ++ mean patch token (768-d for ViT-S/14)."""
import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SRC_H, SRC_W = 224, 398     # SoccerNet 224p frame size
CROP_W = 392                # 28 patches of 14 px; drop 3 px each side, no rescaling


class DinoEncoder:
    def __init__(self, name="dinov2_vits14", device=torch.device("cuda"), batch_frames=256):
        self.device = device
        self.batch = batch_frames
        self.model = torch.hub.load("facebookresearch/dinov2", name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = 2 * self.model.embed_dim
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def _prep(self, frames):
        x = frames.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        if tuple(x.shape[-2:]) != (SRC_H, SRC_W):
            x = F.interpolate(x, size=(SRC_H, SRC_W), mode="bilinear", antialias=True, align_corners=False)
        off = (SRC_W - CROP_W) // 2
        x = x[..., off:off + CROP_W]
        return (x - self.mean) / self.std

    @torch.no_grad()
    def encode(self, frames, flip=False):
        """frames: uint8 tensor (B, H, W, 3) -> float16 CPU tensor (B, dim)."""
        out = []
        amp = self.device.type == "cuda"
        for i in range(0, frames.shape[0], self.batch):
            x = self._prep(frames[i:i + self.batch])
            if flip:
                x = x.flip(-1)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
                f = self.model.forward_features(x)
            feat = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], dim=-1)
            out.append(feat.half().cpu())
        return torch.cat(out)
