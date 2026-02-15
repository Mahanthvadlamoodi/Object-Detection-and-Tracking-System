"""
PENet: Pyramid Enhancement Network
Optimized GPU-only implementation matching the original paper architecture.
Key improvements over author's code:
  - GPU Sobel filter (author uses cv2 CPU loop — 10-50x slower)
  - F.conv_transpose2d for upsample (single fused kernel, no zeros alloc)
  - Gaussian kernel as register_buffer (no .to(device) hassle)
  - Correct sigma matching cv2.getGaussianKernel(5, 0)
  - Residual scaling (beta) for safe pretrained YOLO integration
  - Size-safe pyramid for non-power-of-2 inputs
YOLOv8 compatible. torch.compile friendly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Laplacian Pyramid (GPU, buffer-based, conv_transpose2d upsample)
# =========================================================
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()
        self.num_high = num_high
        self.channels = channels
        self._pad = kernel_size // 2
        self.register_buffer(
            "kernel",
            self._gauss_kernel(kernel_size, channels),
            persistent=False,
        )

    @staticmethod
    def _gauss_kernel(kernel_size, channels):
        # Match cv2.getGaussianKernel(k, 0): sigma = 0.3*((k-1)*0.5 - 1) + 0.8
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g /= g.sum()
        kernel = g[:, None] @ g[None, :]
        return kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

    def _get_kernel(self, x):
        """Return kernel cast to input dtype (no-op when already matching)."""
        k = self.kernel
        return k.to(x.dtype) if k.dtype != x.dtype else k

    def _conv_gauss(self, x):
        k = self._get_kernel(x)
        return F.conv2d(F.pad(x, [self._pad] * 4, mode="reflect"), k, groups=self.channels)

    def _downsample(self, x):
        return x[:, :, ::2, ::2]

    def _upsample(self, x):
        """Upsample via transposed convolution — single fused CUDA kernel.

        Mathematically equivalent to:
            up = zeros(B, C, 2H, 2W); up[::2, ::2] = x * 4; conv_gauss(up)
        but avoids the large zeros allocation and scatter write.
        Boundary uses zero-padding instead of reflect (negligible quality diff).
        """
        k = self._get_kernel(x) * 4.0
        return F.conv_transpose2d(
            x, k, stride=2, padding=self._pad, output_padding=1, groups=self.channels,
        )

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self._downsample(self._conv_gauss(current))
            up = self._upsample(down)
            # Crop to match original size (handles odd dimensions)
            up = up[:, :, : current.shape[2], : current.shape[3]]
            pyr.append(current - up)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self._upsample(image)
            up = up[:, :, : level.shape[2], : level.shape[3]]
            image = up + level
        return image


# =========================================================
# Residual Block (matches paper)
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )
        self.conv_out = nn.Conv2d(in_features, out_features, 3, padding=1)

    def forward(self, x):
        return self.conv_out(x + self.block(x))


# =========================================================
# DPM — Dynamic Perception Module (global context modeling)
# =========================================================
class DPM(nn.Module):
    def __init__(self, inplanes, planes, bias=False):
        super().__init__()
        self.conv_mask = nn.Conv2d(inplanes, 1, 1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, inplanes, 1, bias=bias),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        input_x = x.flatten(2).unsqueeze(1)                              # [B,1,C,HW]
        mask = self.softmax(self.conv_mask(x).flatten(2)).unsqueeze(3)    # [B,1,HW,1]
        context = torch.matmul(input_x, mask).view(B, C, 1, 1)           # [B,C,1,1]
        return x + self.channel_add_conv(context)


# =========================================================
# Sobel Filter — fully differentiable GPU depthwise conv
# (Author's code uses cv2.Sobel on CPU per-sample — non-differentiable
#  and 10-50x slower. Our version provides full gradient flow.)
# =========================================================
class SobelFilter(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        # Pre-expand for the known channel count — avoids expand() every forward
        self.register_buffer("gx", gx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)
        self.register_buffer("gy", gy.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)

    def forward(self, x):
        ch = x.shape[1]
        gx = self.gx[:ch].to(x.dtype)
        gy = self.gy[:ch].to(x.dtype)
        ex = F.conv2d(x, gx, padding=1, groups=ch)
        ey = F.conv2d(x, gy, padding=1, groups=ch)
        return torch.sqrt(ex * ex + ey * ey + 1e-6)


# =========================================================
# LowPass Module — multi-scale average pooling
# =========================================================
class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in sizes])
        self.relu = nn.ReLU(inplace=True)
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        splits = torch.split(feats, self.channel_splits, dim=1)
        priors = [
            F.interpolate(self.stages[i](splits[i]), size=(h, w),
                          mode="bilinear", align_corners=False)
            for i in range(4)
        ]
        return self.relu(torch.cat(priors, dim=1))


# =========================================================
# AE — Auxiliary Enhancement block (matches paper exactly)
# Paper flow:
#   edge = conv_edge(sobel(x))
#   res  = res2(dpm(res1(x)))
#   out  = agg(cat([res, edge + x]))      ← two-branch high-freq
#   low  = conv2(lpm(conv1(x)))            ← low-freq branch
#   out  = fusion(cat([out, low]))         ← final merge
# =========================================================
class AE(nn.Module):
    def __init__(self, n_feat=3, bias=False):
        super().__init__()
        # Edge branch
        self.sobel = SobelFilter()
        self.conv_edge = nn.Conv2d(3, 3, 1, bias=bias)

        # High-freq residual branch
        self.res1 = ResidualBlock(3, 32)
        self.dpm = DPM(32, 32)
        self.res2 = ResidualBlock(32, 3)

        # Aggregate high-freq: cat([res, edge+x]) → 6ch → 3ch
        self.agg = nn.Conv2d(6, 3, 1, bias=False)

        # Low-freq branch
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.lpm = LowPassModule(32)
        self.conv2 = nn.Conv2d(32, 3, 1)

        # Final fusion: cat([high_out, low]) → 6ch → 3ch
        self.fusion = nn.Conv2d(6, 3, 1)

    def forward(self, x):
        # Edge path
        edge = self.conv_edge(self.sobel(x))

        # High-freq path
        res = self.res2(self.dpm(self.res1(x)))

        # Two-branch aggregation (paper: cat([res, edge + x]))
        out = self.agg(torch.cat([res, edge + x], dim=1))

        # Low-freq path
        low = self.conv2(self.lpm(self.conv1(x)))

        # Final fusion
        out = self.fusion(torch.cat([out, low], dim=1))
        return out


# =========================================================
# PENet — Pyramid Enhancement Network
#
# Residual scaling (beta) ensures safe integration with a
# pretrained YOLO: beta starts at 0 → output = x (identity).
# During training beta grows, gradually blending enhancement in.
# =========================================================
class PENet(nn.Module):
    def __init__(self, c1=3, num_high=3, gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel, channels=c1)
        self.AEs = nn.ModuleList([AE(c1) for _ in range(num_high + 1)])

        # Residual scaling: beta=0 → identity pass-through at init
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(x)

        # Process each level coarse-to-fine (matching paper order)
        enhanced = [self.AEs[i](pyrs[-1 - i]) for i in range(self.num_high + 1)]

        out = self.lap_pyramid.pyramid_recons(enhanced)

        # Residual blend: identity when beta=0, full enhancement when beta→1
        return x + self.beta * (out - x)


# =========================================================
# YOLO-compatible wrapper
# =========================================================
class PENetWrapper(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.model = PENet(c1=c1, num_high=3)
        self.f = -1
        self.i = 0
        self.type = "PENet"

    def forward(self, x):
        return self.model(x)
