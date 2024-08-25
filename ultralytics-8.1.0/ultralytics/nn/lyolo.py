import torch
import torch.nn as nn
import math
from einops import rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class MSGConv(nn.Module):
    # Multi-Scale Ghost Conv
    def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        self.s = s
        self.convs = nn.ModuleList([])
        self.cv1 = Conv(c1, min_ch, k, s)
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = Conv(c2, c2, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)
        return x


class MSGAConv(nn.Module):  # MSGRConv
    # Multi-Scale Ghost Residual Conv
    def __init__(self, c1, c2, k=3, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        self.s = s

        self.convs = nn.ModuleList([])
        if s == 1:
            self.cv1 = Conv(c1, min_ch, 1, 1)
        if s == 2:
            self.cv1 = Conv(c1, min_ch, 3, 2)
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = Conv(c2, c2, 1)
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):

        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)

        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        out = torch.cat([x1, x2], dim=1)
        x = self.shortcut(x)
        out = self.conv1x1(out) + x
        return out


class MSGBottleneck(nn.Module):
    # GhostBottleneck  Conv->MSGConv
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(MSGConv(c1, c_),
                                  MSGConv(c_, c2))
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MSGABottleneck(nn.Module):  # MSGRBottleneck

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(MSGAConv(c1, c_),
                                  MSGAConv(c_, c2))
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3MSGR(nn.Module):  # C3MSGR

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(MSGABottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C2fMSGR(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MSGABottleneck(self.c, self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MSGELAN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = MSGBottleneck(c_, 2 * c_)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)

        return self.cv3(torch.cat((x1, x2, x3), 1))


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)
        W = self.enc(W)
        W = self.pix_shf(W)
        W = torch.softmax(W, dim=1)

        X = self.upsmp(X)
        X = self.unfold(X)
        X = X.view(b, c, -1, h_, w_)

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])
        return X
