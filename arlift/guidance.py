"""Spatial guidance for the two autoregressive backbones (BTCHW latents)."""
import math


def gaussian_parameters(sigma, kernel_size):
    if (type(sigma) not in (float, int) or not math.isfinite(sigma) or sigma <= 0
            or type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1):
        raise ValueError('Gaussian requires finite positive sigma and an odd integer kernel')
    return sigma, kernel_size


def spatial_gaussian(value, sigma, kernel_size):
    import torch
    import torch.nn.functional as F
    gaussian_parameters(sigma, kernel_size)
    if value.ndim != 5:
        raise ValueError('Expected normalized latent layout [B,T,C,H,W]')
    b, t, c, h, w = value.shape
    radius = kernel_size // 2
    if min(h, w) <= radius:
        raise ValueError('Reflect padding requires height/width greater than kernel radius')
    # Disabling autocast keeps subtraction, kernel and both convolutions in FP32.
    with torch.autocast(device_type=value.device.type, enabled=False):
        coords = torch.arange(-radius, radius + 1, device=value.device, dtype=torch.float32)
        kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        planes = value.float().reshape(b * t * c, 1, h, w)
        result = F.conv2d(F.pad(planes, (radius, radius, 0, 0), mode='reflect'),
                          kernel.view(1, 1, 1, -1))
        result = F.conv2d(F.pad(result, (0, 0, radius, radius), mode='reflect'),
                          kernel.view(1, 1, -1, 1))
    return result.reshape(b, t, c, h, w)


def resize_reference(clean, target_hw):
    import torch.nn.functional as F
    if clean.ndim != 5 or tuple(target_hw) != tuple(2 * x for x in clean.shape[-2:]):
        raise ValueError('Reference bridge requires a spatial 2x normalized BTCHW latent')
    # Same dtype/normalization/amplitude. Never decode, encode or re-noise a reference.
    return F.interpolate(clean.flatten(0, 1), size=tuple(target_hw), mode='bilinear',
                         align_corners=False).unflatten(0, clean.shape[:2])


def fuse_clean(prediction, reference, weight, sigma, kernel_size):
    gaussian_parameters(sigma, kernel_size)
    if type(weight) not in (float, int) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Lambda must be finite and in [0,1]')
    if (prediction.shape != reference.shape or prediction.dtype != reference.dtype
            or prediction.device != reference.device or prediction.ndim != 5):
        raise ValueError('Reference and clean prediction must share BTCHW geometry/dtype/device')
    if weight == 0:
        return prediction
    import torch
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        residual = reference.float() - prediction.float()
        correction = spatial_gaussian(residual, sigma, kernel_size)
        result = (prediction.float() + weight * correction).to(prediction.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite guided prediction; no clipping or fallback')
    return result
