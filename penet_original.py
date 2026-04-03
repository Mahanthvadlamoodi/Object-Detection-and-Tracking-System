import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Dynamic Edge Enhancer (Replaces slow CPU cv2.sobel)
class DynamicEdgeEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        # Initialize dynamically with Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        sobel = (sobel_x + sobel_y) / 2.0
        self.edge.weight.data = sobel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def forward(self, x):
        return self.edge(x)


# Tone Curve (For LF component enhancement)
class ToneCurve(nn.Module):
    def __init__(self, n_iter=4):
        super().__init__()
        self.n_iter = n_iter
        self.pred = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 3*n_iter, 1),
            nn.Tanh()
        )

    def forward(self, x):
        alphas = self.pred(x)
        out = x
        for i in range(self.n_iter):
            a = alphas[:, i*3:(i+1)*3]
            out = out + a * out * (1 - out)
        return out


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()
        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)
        
        # Gated fusion dynamic learnable parameters
        self.w1 = nn.Parameter(torch.ones(num_high, 1, 1, 1))
        self.w2 = nn.Parameter(torch.ones(num_high, 1, 1, 1))

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        kernel = kernel.to(dtype=x.dtype, device=x.device)
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         dtype=x.dtype, device=x.device)
        up[:, :, ::2, ::2] = x * 4
        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current) # Lowest frequency level
        return pyr

    def pyramid_recons(self, pyr):
        # Reconstruct sequentially from bottom (LF) up to top (HF)
        # pyr[-1]: base LF, pyr[0]: highest frequency details
        image = pyr[-1]
        for i in reversed(range(self.num_high)):
            up = self.upsample(image)
            # Dynamic learnable gated fusion!
            image = self.w1[i] * up + self.w2[i] * pyr[i]
        return image


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_x = nn.Conv2d(in_features, out_features, 3, padding=1)

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_x(x + self.block(x))


class DPM(nn.Module):
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), bias=False):
        super(DPM, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x.view(batch, channel, height * width)
        context_mask = self.conv_mask(x).view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        # BMM efficiently prevents the PyTorch 32GB memory cache leak
        context = torch.bmm(input_x, context_mask.transpose(1, 2)).view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        channel_add_term = self.channel_add_conv(context)
        return x + channel_add_term


class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats_split = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.interpolate(self.stages[i](feats_split[i]), size=(h, w), mode='bilinear', align_corners=False) for i in range(4)]
        bottle = torch.cat(priors, 1)
        return self.relu(bottle)


class AE(nn.Module):
    def __init__(self, n_feat=3, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), groups=1):
        super(AE, self).__init__()

        self.agg = nn.Conv2d(6, 3, 1, stride=1, padding=0, bias=False)
        self.edge_extractor = DynamicEdgeEnhancer()
        self.conv_edge = nn.Conv2d(3, 3, kernel_size=1, bias=bias)

        # Restoring these full blocks returns the ~90k parameter count smoothly
        self.res1 = ResidualBlock(3, 32)
        self.res2 = ResidualBlock(32, 3)
        self.dpm = nn.Sequential(DPM(32, 32))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.low_pass = LowPassModule(32)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1)
        self.low_tone = ToneCurve() # LF Enhancement Module
        
        self.fusion = nn.Conv2d(6, 3, kernel_size=1)

    def forward(self, x):
        # 1. Edge Enhancement Path 
        s_x = self.edge_extractor(x)
        s_x = self.conv_edge(s_x)

        # 2. Main High-Freq Path (with memory-safe DPM)
        res = self.res1(x)
        res = self.dpm(res)
        res = self.res2(res)
        
        out = torch.cat([res, s_x + x], dim=1)
        out = self.agg(out)

        # 3. Dynamic Low Frequency Path (with Tone Curve integration)
        low_fea = self.conv1(x)
        low_fea = self.low_pass(low_fea)
        low_fea = self.conv2(low_fea)
        low_fea = self.low_tone(low_fea)

        out = torch.cat([out, low_fea], dim=1)
        out = self.fusion(out)

        return out


class PENetFinal(nn.Module):
    def __init__(self, num_high=3, gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.aes = nn.ModuleList([AE(3) for _ in range(num_high + 1)])

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
        for i, level in enumerate(pyrs):
            trans_pyr = self.aes[i](level)
            trans_pyrs.append(trans_pyr)

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)
        return torch.sigmoid(out)


class PENetWrapper(PENetFinal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = -1
        self.i = 0
        self.type = "PENet"

    def forward(self, x):
        self.last_input = x
        return super().forward(x)

__all__ = ["PENetWrapper"]