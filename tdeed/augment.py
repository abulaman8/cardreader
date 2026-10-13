"""T-DEED clip-level augmentation on GPU tensors + mixup. One random transform per clip (same for all frames)."""
import torch
import torchvision.transforms.functional as TF


def augment_clip(x, rng, a):
    """x float (T, 3, H, W) in [0, 1] -> same. rng: numpy Generator. a: cfg['augment']."""
    if rng.random() < a["p_hue"]:
        x = TF.adjust_hue(x, float(rng.uniform(-a["hue"], a["hue"])))
    if rng.random() < a["p_saturation"]:
        x = TF.adjust_saturation(x, float(rng.uniform(*a["saturation"])))
    if rng.random() < a["p_brightness"]:
        x = TF.adjust_brightness(x, float(rng.uniform(*a["brightness"])))
    if rng.random() < a["p_contrast"]:
        x = TF.adjust_contrast(x, float(rng.uniform(*a["contrast"])))
    if rng.random() < a["p_blur"]:
        s = float(rng.uniform(0.1, 2.0))
        k = int(a["blur_kernel"])
        x = TF.gaussian_blur(x, [k, k], [s, s])
    if rng.random() < a["hflip"]:
        x = x.flip(-1)
    return x.clamp_(0.0, 1.0)


def augment_batch(x, rng, a):
    """x float (B, T, 3, H, W) in [0, 1]."""
    return torch.stack([augment_clip(x[b], rng, a) for b in range(x.shape[0])])


def mixup(x, y, dt, m, alpha, rng):
    """T-DEED mixup with an in-batch partner. y: soft class targets (B, T, C); dt displacement; m loss mask."""
    B = x.shape[0]
    lam = torch.as_tensor(rng.beta(alpha, alpha, B), device=x.device, dtype=torch.float32)
    perm = torch.as_tensor(rng.permutation(B), device=x.device)
    lx = lam.view(B, 1, 1, 1, 1).to(x.dtype)
    x = lx * x + (1 - lx) * x[perm]
    lt = lam.view(B, 1)
    y = lt[..., None] * y + (1 - lt[..., None]) * y[perm]
    dt = lt * dt + (1 - lt) * dt[perm]
    m = m * m[perm]
    return x, y, dt, m
