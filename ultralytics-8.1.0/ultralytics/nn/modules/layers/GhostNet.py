#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   GhostNet.py
@Time      :   2024/02/26 20:15:41
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn

__all__ = "G_bneck"


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module(
            "FC1", nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        )  # 1*1卷积与此效果相同
        self.Excitation.add_module("ReLU", nn.ReLU())
        self.Excitation.add_module(
            "FC2", nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        )
        self.Excitation.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
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
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GhostConv(nn.Module):
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class G_bneck(nn.Module):
    def __init__(
        self, c1, c2, midc, k=5, s=1, use_se=False
    ):  # ch_in, ch_mid, ch_out, kernel, stride, use_se
        super().__init__()
        assert s in [1, 2]
        c_ = midc
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            Conv(c_, c_, 3, s=2, p=1, g=c_, act=False) if s == 2 else nn.Identity(),
            SeBlock(c_) if use_se else nn.Sequential(),
            GhostConv(c_, c2, 1, 1, act=False),
        )

        self.shortcut = (
            nn.Identity()
            if (c1 == c2 and s == 1)
            else nn.Sequential(
                Conv(c1, c1, 3, s=s, p=1, g=c1, act=False),
                Conv(c1, c2, 1, 1, act=False),
            )
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
