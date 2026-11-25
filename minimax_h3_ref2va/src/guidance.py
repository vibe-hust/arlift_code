"""Apply a spatial Gaussian correction to the clean video prediction."""
import torch
from torch.nn import functional as F


class GaussianVideoGuide:
    def __init__(self, clean_lr, target_hw, gaussian):
        self.reference = F.interpolate(
            clean_lr.flatten(0, 1), size=target_hw, mode="bilinear", align_corners=False
        ).unflatten(0, clean_lr.shape[:2])
        self.radius = gaussian["kernel_size"] // 2
        positions = torch.arange(
            -self.radius, self.radius + 1, dtype=torch.float32, device=clean_lr.device
        )
        kernel = torch.exp(-positions.square() / (2 * gaussian["sigma"] ** 2))
        self.kernel = kernel / kernel.sum()

    def fuse(self, video_clean, fusion_weight):
        if fusion_weight == 0.0:
            return video_clean
        if video_clean.shape != self.reference.shape or video_clean.dtype != torch.float32:
            raise RuntimeError("Expected a normalized FP32 video clean prediction matching the reference.")

        radius = self.radius
        with torch.autocast(device_type=video_clean.device.type, enabled=False):
            planes = (self.reference - video_clean).flatten(0, 2).unsqueeze(1)
            horizontal = F.conv2d(
                F.pad(planes, (radius, radius, 0, 0), mode="reflect"),
                self.kernel.view(1, 1, 1, -1),
            )
            filtered = F.conv2d(
                F.pad(horizontal, (0, 0, radius, radius), mode="reflect"),
                self.kernel.view(1, 1, -1, 1),
            )
            video_clean = video_clean + fusion_weight * filtered.reshape_as(video_clean)
        return video_clean
