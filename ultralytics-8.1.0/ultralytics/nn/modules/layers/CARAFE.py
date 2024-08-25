#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   CARAFE.py
@Time      :   2024/02/26 20:13:57
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch.nn.functional as F
import torch
from torch import nn

__all__ = "CARAFE"


class CARAFE(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(
            c1 // 4,
            self.up_factor**2 * self.kernel_size**2,
            self.kernel_size,
            1,
            self.kernel_size // 2,
        )
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(
            kernel_tensor, self.up_factor
        )  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(
            2, self.up_factor, step=self.up_factor
        )  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(
            3, self.up_factor, step=self.up_factor
        )  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(
            N, self.kernel_size**2, H, W, self.up_factor**2
        )  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(
            x,
            pad=(
                self.kernel_size // 2,
                self.kernel_size // 2,
                self.kernel_size // 2,
                self.kernel_size // 2,
            ),
            mode="constant",
            value=0,
        )  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor
