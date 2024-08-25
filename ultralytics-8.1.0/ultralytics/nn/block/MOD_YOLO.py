import torch
from torch import nn
import warnings


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class MODSConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x1 = self.dconv(x) + x
        return self.pconv(x1)


class GRF_SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        Conv = BaseConv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 6, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.am = nn.AdaptiveMaxPool2d(1)
        self.aa = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2), self.am(x).expand_as(x), self.aa(x).expand_as(x)), 1))


class DAF_CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(DAF_CA, self).__init__()
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=(1, 2), stride=1, padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, bias=False)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        pool_ha = nn.AdaptiveAvgPool2d((h, 1))
        pool_hm = nn.AdaptiveMaxPool2d((h, 1))
        x_ha = pool_ha(x)
        x_hm = pool_hm(x)
        x_h = torch.cat([x_ha, x_hm], dim=3)
        pool_wa = nn.AdaptiveAvgPool2d((1, w))
        pool_wm = nn.AdaptiveMaxPool2d((1, w))
        x_wa = pool_wa(x).permute(0, 1, 3, 2)
        x_wm = pool_wm(x).permute(0, 1, 3, 2)
        x_w = torch.cat([x_wa, x_wm], dim=3)
        y1 = torch.cat([x_h, x_w], dim=2)
        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y_h, y_w = torch.split(y1, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = (self.conv_h(y_h)).sigmoid()
        a_w = (self.conv_w(y_w)).sigmoid()

        return identity * a_h * a_w


class MODSLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, act="silu"):
        super().__init__()
        self.s = BaseConv(in_channels, out_channels, ksize=1, stride=1, act=act)
        self.conv3 = DAF_CA(out_channels, out_channels, reduction=32)
        module_list = [MODSConv(in_channels, in_channels, ksize=3) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.m(x)
        x = x + x_1
        return self.conv3(self.s(x))


class MODL_Head(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu"):
        super().__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.cls_convs.append(nn.Sequential(*[
                MODSConv(in_channels=int(in_channels[i] * width), out_channels=int(in_channels[i] * width), ksize=3,
                         stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(in_channels[i] * width), out_channels=num_classes, kernel_size=1, stride=1,
                          padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                MODSConv(in_channels=int(in_channels[i] * width), out_channels=int(in_channels[i] * width), ksize=3,
                         stride=1, act=act),
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(in_channels[i] * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(in_channels[i] * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs