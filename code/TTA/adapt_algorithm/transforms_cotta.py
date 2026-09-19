import torch
import torch.nn.functional as F


def gaussian_noise(x, std=0.01):
    """添加高斯噪声"""
    noise = torch.randn_like(x) * std
    return x + noise


def sliding_window_smoothing(x, window_size=3):
    """滑动窗口平滑（每个通道分别）"""
    B, _, T, C = x.shape
    pad = window_size // 2
    x_padded = F.pad(x, (0, 0, pad, pad), mode='reflect')  # pad time dimension
    x_smooth = torch.zeros_like(x)
    for i in range(window_size):
        x_smooth += x_padded[:, :, i:i + T, :]
    x_smooth /= window_size
    return x_smooth


def time_jitter(x, max_jitter=2):
    """时间抖动（向前或向后移动），边界补零"""
    B, _, T, C = x.shape
    x_jittered = torch.zeros_like(x)
    for b in range(B):
        shift = torch.randint(-max_jitter, max_jitter + 1, (1,)).item()
        if shift > 0:
            x_jittered[b, :, shift:, :] = x[b, :, :-shift, :]
        elif shift < 0:
            x_jittered[b, :, :shift, :] = x[b, :, -shift:, :]
        else:
            x_jittered[b] = x[b]
    return x_jittered


def feature_scaling(x, scale_range=(0.9, 1.1)):
    """每个通道乘以缩放因子"""
    B, _, T, C = x.shape
    scales = torch.empty((B, 1, 1, C), device=x.device).uniform_(*scale_range)
    return x * scales


import torch.nn.functional as F

def time_crop_fill(x: torch.Tensor, crop_ratio=0.9) -> torch.Tensor:
    """
    对时间序列进行裁剪并通过插值填补回来。
    保证输出形状与输入一致：(B, 1, T, C)
    """
    B, _, T, C = x.shape
    crop_len = int(T * crop_ratio)
    start = torch.randint(0, T - crop_len + 1, (B,))
    x_new = torch.zeros_like(x)

    for b in range(B):
        crop = x[b, 0, start[b]:start[b] + crop_len, :].transpose(0, 1).unsqueeze(0)  # (1, C, crop_len)
        crop_interp = F.interpolate(crop, size=T, mode='linear', align_corners=True)
        x_new[b, 0] = crop_interp.squeeze(0).transpose(0, 1)  # (C, T) → (T, C)

    return x_new
def get_tta_transforms(x: torch.Tensor, mode: str = "all", noise_std=0.01) -> torch.Tensor:
    """
    输入：x of shape (B, 1, T, C)，类型为 torch.Tensor
    输出：增强后的 x_aug，仍然是 (B, 1, T, C)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected input to be torch.Tensor")

    if mode == "noise":
        return gaussian_noise(x, std=noise_std)
    elif mode == "smooth":
        return sliding_window_smoothing(x)
    elif mode == "jitter":
        return time_jitter(x)
    elif mode == "scale":
        return feature_scaling(x)
    elif mode == "crop":
        return time_crop_fill(x)
    elif mode == "all":
        # 串行应用所有增强，顺序可调
        x_aug = gaussian_noise(x, std=noise_std)
        x_aug = time_crop_fill(x_aug)
        x_aug = time_jitter(x_aug)

        return x_aug
    else:
        raise ValueError(f"Unsupported mode: {mode}")


