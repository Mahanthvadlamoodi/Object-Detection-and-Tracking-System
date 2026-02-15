"""
PENet v2: Pyramid Enhancement Network for Dark Object Detection
================================================================
Faithful to PE-YOLO paper (Liang et al., ICANN 2023) with targeted improvements:

  1. GPU-native Sobel (author uses cv2 CPU loop per sample — 10-50x slower)
  2. Learnable Adaptive Fusion for LF/HF features in AE blocks
     (author does simple concat + 1x1 conv with equal weight)
  3. F.conv_transpose2d upsample (fused kernel, no zeros alloc)
  4. register_buffer for Gaussian kernel (auto device/dtype)
  5. No beta residual scaling — PENet output goes directly to YOLO
     (matching author's approach: PENet IS the enhancement, not a residual)

Architecture (matching paper):
  Image → Laplacian Pyramid → [HF_0, HF_1, HF_2, LF]
                                 ↓      ↓      ↓    ↓
                               AE_3   AE_2   AE_1  AE_0
                                 ↓      ↓      ↓    ↓
                            Pyramid Reconstruction → Enhanced Image (3ch)

Each AE block:
  ├─ Edge branch:    Sobel(x) → conv_edge
  ├─ HF branch:      ResidualBlock(x) → DPM → ResidualBlock → res
  ├─ Stage 1:        agg(cat([res, edge + x])) → hf_out
  ├─ LF branch:      conv1(x) → LowPassModule → conv2 → lf_out
  └─ Stage 2:        AdaptiveFusion(hf_out, lf_out, x) → out    ← NEW: learnable fusion
"""

import math
import cv2
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

        # Gaussian kernel matching cv2.getGaussianKernel(k, 0)
        self.register_buffer(
            "kernel",
            self._gauss_kernel(kernel_size, channels),
            persistent=False,
        )

    @staticmethod
    def _gauss_kernel(kernel_size, channels):
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g /= g.sum()
        kernel = g[:, None] @ g[None, :]
        return kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

    def _get_kernel(self, x):
        k = self.kernel
        return k.to(x.dtype) if k.dtype != x.dtype else k

    def _conv_gauss(self, x):
        k = self._get_kernel(x)
        return F.conv2d(F.pad(x, [self._pad] * 4, mode="reflect"), k, groups=self.channels)

    def _downsample(self, x):
        return x[:, :, ::2, ::2]

    def _upsample(self, x):
        """Upsample via transposed convolution — single fused CUDA kernel."""
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
            up = up[:, :, :current.shape[2], :current.shape[3]]
            pyr.append(current - up)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self._upsample(image)
            up = up[:, :, :level.shape[2], :level.shape[3]]
            image = up + level
        return image


# =========================================================
# Sobel Filter — GPU depthwise conv (replaces author's CPU cv2 loop)
# =========================================================
class SobelFilter(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("gx", sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)
        self.register_buffer("gy", sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)
        self.channels = channels

    def forward(self, x):
        ch = x.shape[1]
        gx = self.gx[:ch].to(x.dtype)
        gy = self.gy[:ch].to(x.dtype)
        ex = F.conv2d(x, gx, padding=1, groups=ch)
        ey = F.conv2d(x, gy, padding=1, groups=ch)
        # Author uses: addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        return 0.5 * ex + 0.5 * ey


# =========================================================
# Residual Block (matches author exactly)
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_x = nn.Conv2d(in_features, out_features, 3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_x(x + self.block(x))


# =========================================================
# DPM — Dynamic Processing Module (matches author exactly)
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
        input_x = x.flatten(2).unsqueeze(1)                            # [B,1,C,HW]
        mask = self.softmax(self.conv_mask(x).flatten(2)).unsqueeze(3)  # [B,1,HW,1]
        context = torch.matmul(input_x, mask).view(B, C, 1, 1)         # [B,C,1,1]
        return x + self.channel_add_conv(context)


# =========================================================
# LowPass Module — multi-scale average pooling (matches author)
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
# Adaptive Fusion — learnable attention for HF/LF merging
# Instead of author's simple concat+1x1: cat([hf, lf]) → Conv1x1
# We learn per-pixel importance: alpha * hf + (1-alpha) * lf
# where alpha is conditioned on the original input image
# =========================================================
class AdaptiveFusion(nn.Module):
    """
    Learns spatially-varying attention map to fuse HF and LF features.
    
    Given:
      hf: high-frequency enhanced features (3ch)
      lf: low-frequency enhanced features (3ch)  
      x:  original input (3ch) — provides context for the gate
      
    Output: alpha * hf + (1-alpha) * lf
    where alpha = sigmoid(gate(cat([hf, lf, x])))
    """
    def __init__(self, channels=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Initialize gate bias to 0 → sigmoid(0) = 0.5 → equal weighting at start
        nn.init.zeros_(self.gate[-2].bias)

    def forward(self, hf, lf, x):
        alpha = self.gate(torch.cat([hf, lf, x], dim=1))
        return alpha * hf + (1.0 - alpha) * lf


# =========================================================
# AE — Appearance Enhancement block
# Matches author's structure exactly + AdaptiveFusion improvement
# =========================================================
class AE(nn.Module):
    def __init__(self, n_feat=3, bias=False):
        super().__init__()
        # Edge branch (GPU Sobel replaces author's CPU cv2 loop)
        self.sobel = SobelFilter(channels=n_feat)
        self.conv_edge = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        # High-frequency: residual + DPM
        self.res1 = ResidualBlock(n_feat, 32)
        self.dpm = DPM(32, 32)
        self.res2 = ResidualBlock(32, n_feat)

        # Stage 1 aggregation: cat([res, edge+x]) → 6ch → 3ch
        self.agg = nn.Conv2d(n_feat * 2, n_feat, 1, stride=1, padding=0, bias=False)

        # Low-frequency branch
        self.conv1 = nn.Conv2d(n_feat, 32, kernel_size=1)
        self.lpm = LowPassModule(32)
        self.conv2 = nn.Conv2d(32, n_feat, kernel_size=1)

        # Stage 2: Adaptive fusion (our improvement over author's simple concat)
        self.adaptive_fusion = AdaptiveFusion(channels=n_feat)

    def forward(self, x):
        # Edge detection
        edge = self.conv_edge(self.sobel(x))

        # High-frequency path
        res = self.res1(x)
        res = self.dpm(res)
        res = self.res2(res)

        # Stage 1: aggregate HF features
        hf = self.agg(torch.cat([res, edge + x], dim=1))

        # Low-frequency path
        lf = self.conv1(x)
        lf = self.lpm(lf)
        lf = self.conv2(lf)

        # Stage 2: learnable fusion (replaces author's concat + 1x1 conv)
        out = self.adaptive_fusion(hf, lf, x)
        return out


# =========================================================
# PENet — Pyramid Enhancement Network
# Matches paper exactly: decompose → enhance each level → reconstruct
# No beta residual scaling (author doesn't use it)
# =========================================================
class PENet(nn.Module):
    def __init__(self, c1=3, num_high=3, gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel, channels=c1)

        # One AE per pyramid level (num_high HF levels + 1 LF level)
        self.AEs = nn.ModuleList([AE(c1) for _ in range(num_high + 1)])

    def forward(self, x):
        # Decompose into Laplacian pyramid
        pyrs = self.lap_pyramid.pyramid_decom(x)

        # Enhance each level (coarse-to-fine, matching author's order)
        trans_pyrs = []
        for i in range(self.num_high + 1):
            trans_pyr = self.AEs[i](pyrs[-1 - i])
            trans_pyrs.append(trans_pyr)

        # Reconstruct enhanced image from pyramid
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)
        return out


# =========================================================
# YOLO-compatible Wrapper
# Outputs 3-channel enhanced image (same as author)
# No beta — PENet must learn proper enhancement directly
# =========================================================
class PENetWrapper(nn.Module):
    """
    YOLO integration wrapper for PENet.
    
    Input:  3ch image [B, 3, H, W]
    Output: 3ch enhanced image [B, 3, H, W]
    
    Usage in YAML:
      - [-1, 1, PENetWrapper, [3]]  # c2=3 (output channels)
    """
    def __init__(self, c1=3, c2=3, num_high=3):
        super().__init__()
        self.model = PENet(c1=c1, num_high=num_high)

        # YOLO integration attributes
        self.f = -1
        self.i = 0
        self.type = "PENet_v2.PENetWrapper"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.model(x)
