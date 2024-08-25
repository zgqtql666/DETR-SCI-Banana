#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   DRepConv.py
@Time      :   2024/02/26 20:14:51
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


from torch import nn
import torch
import math
import torch.nn.functional as F

__all__ = "DRepConv"


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


class DRepConv(nn.Module):
    """
    Dilated Re-param Conv proposed in UniRepLKNet (https://arxiv.org/pdf/2311.15599.pdf)
    This module is used in UniRepLKNet.
    Based on https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py
    """

    default_act = nn.SiLU()  # default activation
    SETTING_CONFIG = {
        # kernel dilates and stride in different large kernel size
        17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
        15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
        9: ([5, 5, 3, 3], [1, 2, 3, 4]),
        7: ([5, 3, 3], [1, 2, 3]),
        5: ([3, 3], [1, 2]),
    }

    def __init__(self, c1, c2, k=1, act=True):
        super().__init__()

        self.cv1 = Conv(c1, c2, k, g=math.gcd(c1, c2), act=False)
        self.kernel_sizes, self.dilates = (
            self.SETTING_CONFIG[k][0],
            self.SETTING_CONFIG[k][1],
        )
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__setattr__(
                f"cv_k{k}_r{r}",
                Conv(
                    c1,
                    c2,
                    k=k,
                    s=1,
                    p=(r * (k - 1) + 1) // 2,
                    g=math.gcd(c1, c2),
                    d=r,
                    act=False,
                ),
            )

    def forward(self, x):
        out = self.cv1(x)
        for k, r in zip(self.kernel_sizes, self.dilates):
            out = out + self.__getattr__(f"cv_k{k}_r{r}")(x)
        return self.act(out)

    def forward_fuse(self, x):
        return self.act(self.cv1(x))

    def fuse_convs(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cv1 = nn.Conv2d(
            in_channels=self.cv1.conv.in_channels,
            out_channels=self.cv1.conv.out_channels,
            kernel_size=self.cv1.conv.kernel_size,
            stride=self.cv1.conv.stride,
            padding=self.cv1.conv.padding,
            dilation=self.cv1.conv.dilation,
            groups=self.cv1.conv.groups,
            bias=True,
        ).requires_grad_(False)

        self.cv1.weight.data = kernel
        self.cv1.bias.data = bias

    def get_equivalent_kernel_bias(self):
        cv1_w, cv1_b = self._fuse_bn(self.cv1.conv, self.cv1.bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            cv_k_r = self.__getattr__(f"cv_k{k}_r{r}")
            cv_k_r_w, cv_k_r_b = self._fuse_bn(cv_k_r.conv, cv_k_r.bn)
            cv1_w = self.merge_dilated_into_large_kernel(cv1_w, cv_k_r_w, r)
            cv1_b += cv_k_r_b
            self.__delattr__(f"cv_k{k}_r{r}")
        return cv1_w, cv1_b

    @staticmethod
    def _fuse_bn(conv, bn):
        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

        # Prepare spatial bias
        b_conv = (
            torch.zeros(conv.weight.size(0), device=conv.weight.device)
            if conv.bias is None
            else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        return (
            torch.mm(w_bn, w_conv).view(conv.weight.shape),
            torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn,
        )

    def convert_dilated_to_nondilated(self, kernel, dilate_rate):
        identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)

        if self.is_depthwise_kernel(kernel):
            return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        else:
            return self.convert_dense_to_nondilated(kernel, dilate_rate)

    @staticmethod
    def is_depthwise_kernel(kernel):
        return kernel.size(1) == 1

    @staticmethod
    def convert_dense_to_nondilated(kernel, dilate_rate):
        slices = [
            F.conv_transpose2d(
                kernel[:, i : i + 1, :, :],
                torch.ones((1, 1, 1, 1)).to(kernel.device),
                stride=dilate_rate,
            )
            for i in range(kernel.size(1))
        ]
        return torch.cat(slices, dim=1)

    def merge_dilated_into_large_kernel(
        self, large_kernel, dilated_kernel, dilate_rate
    ):
        large_kernel_size = large_kernel.size(2)
        dilated_kernel_size = dilated_kernel.size(2)
        equivalent_kernel_size = dilate_rate * (dilated_kernel_size - 1) + 1

        equivalent_kernel = self.convert_dilated_to_nondilated(
            dilated_kernel, dilate_rate
        )
        rows_to_pad = large_kernel_size // 2 - equivalent_kernel_size // 2

        merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
        return merged_kernel
