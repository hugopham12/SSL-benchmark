import torch
import torch.nn.functional as F
import random

# --- Individual augmentations ---

def history_cutout(x, size=8, p=0.5):
    """
    Randomly mask out a contiguous chunk of the sequence without affecting the last step.
    Args:
        x (Tensor): (L, D)
        size (int): size of the cutout window
        p (float): probability of applying
    """
    if random.random() > p:
        return x
    L, D = x.shape
    if L - size - 1 <= 0:
        return x
    start = random.randint(0, L - size - 1)
    mask = torch.ones((L,), device=x.device)
    mask[start:start + size] = 0
    mask[-1] = 1  # always keep the last timestep
    mask = mask.view(-1, 1)
    return x * mask

def history_crop(x, min_ratio=0.5, p=0.5):
    """
    Randomly crop the beginning of the sequence (without touching the last step).
    Args:
        x (Tensor): (L, D)
        min_ratio (float): minimum ratio of history to preserve
        p (float): probability of applying
    """
    if random.random() > p:
        return x
    L, D = x.shape
    min_crop = int(L * (1 - min_ratio))
    if min_crop >= L - 1:
        return x
    start = random.randint(0, min_crop)
    cropped = x[start:]
    pad_size = start
    padding = torch.zeros((pad_size, D), device=x.device)
    out = torch.cat([padding, cropped], dim=0)
    return out

def add_normal_bias(x, std=0.1):
    """
    Add Gaussian noise to each non-padding element.
    Args:
        x (Tensor): (L, D)
        std (float): standard deviation of the noise
    """
    mask = (x != 0).float()
    noise = torch.randn_like(x) * std
    return x + mask * noise

def spatial_dropout(x, rate=0.1):
    """
    Randomly drop entire features (channels) across the sequence.
    Args:
        x (Tensor): (L, D)
        rate (float): dropout rate
    """
    L, D = x.shape
    drop_mask = (torch.rand((1, D), device=x.device) > rate).float()
    return x * drop_mask

# --- Compose augmentations ---

def apply_augmentations(x):
    """
    Apply the sequence of augmentations defined for NCL.
    """
    x = history_cutout(x)
    x = history_crop(x)
    x = add_normal_bias(x)
    x = spatial_dropout(x)
    return x

# --- Main augment_fn ---

def augment_fn(x):
    """
    Given a sequence x, return two independently augmented views.
    Args:
        x (Tensor): (L, D)
    Returns:
        x1, x2 (Tensor): two augmented versions of x
    """
    x1 = apply_augmentations(x)
    x2 = apply_augmentations(x)
    return x1, x2
