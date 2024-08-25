#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   PPLCNet.py
@Time      :   2024/02/26 20:16:56
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn

__all__ = "DepthSepConv"


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module(
            "FC1", nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        )
        self.Excitation.add_module("ReLU", nn.ReLU())
        self.Excitation.add_module(
            "FC2", nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        )
        self.Excitation.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class DepthSepConv(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se):
        super(DepthSepConv, self).__init__()
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dw_size = dw_size
        self.dw_sp = nn.Sequential(
            nn.Conv2d(
                self.inp,
                self.inp,
                kernel_size=self.dw_size,
                stride=self.stride,
                padding=(dw_size - 1) // 2,
                groups=self.inp,
                bias=False,
            ),
            nn.BatchNorm2d(self.inp),
            nn.Hardswish(),
            SeBlock(self.inp, reduction=16) if use_se else nn.Sequential(),
            nn.Conv2d(
                self.inp, self.oup, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.oup),
            nn.Hardswish(),
        )

    def forward(self, x):
        y = self.dw_sp(x)
        return y
