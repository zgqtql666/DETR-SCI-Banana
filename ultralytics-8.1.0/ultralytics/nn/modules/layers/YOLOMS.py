#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   YOLOMS.py
@Time      :   2024/03/05 20:43:19
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


from typing import Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ("MSBlock", "C2f_MSBlock")


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MSBlockLayer(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: Union[int, Sequence[int]]
    ):
        super().__init__()
        self.in_conv = Conv(in_channel, out_channel, 1)
        self.mid_conv = Conv(out_channel, out_channel, k=kernel_size, g=out_channel)
        self.out_conv = Conv(out_channel, in_channel, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x = self.in_conv(x)
        x = self.mid_conv(x)
        x = self.out_conv(x)
        return x


class MSBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_sizes: Sequence[Union[int, Sequence[int]]],
        in_expand_ratio: float = 3.0,
        mid_expand_ratio: float = 2.0,
        layers_num: int = 3,
        in_down_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.in_channel = int(in_channel * in_expand_ratio // in_down_ratio)
        self.mid_channel = self.in_channel // len(kernel_sizes)
        # self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel * mid_expand_ratio)
        self.layers_num = layers_num
        # self.in_attention = None

        self.in_conv = Conv(in_channel, self.in_channel, 1)

        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [
                MSBlockLayer(
                    self.mid_channel,
                    groups,
                    kernel_size=kernel_size,
                )
                for _ in range(int(self.layers_num))
            ]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(self.in_channel, out_channel, 1)
        self.attention = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel : (i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


class C2f_MSBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            MSBlock(self.c, self.c, kernel_sizes=[1, 3, 3]) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
