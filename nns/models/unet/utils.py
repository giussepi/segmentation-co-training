# -*- coding: utf-8 -*-
"""
nns/models/unet/utils

Source: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/utils.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from nns.models.unet.network_others import init_weights


__all__ = [
    'unetConv2', 'UnetGridGatingSignal',  'unetUp', 'UnetUp_CT', 'UnetDsv', 'UnetConv3',
    'UnetUp3_CT'
]


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.n = n
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class UnetGridGatingSignal(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, is_batchnorm=True):
        super().__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=0),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super().__init__()
        # self.conv = unetConv2(in_size, out_size, False)  # original
        self.conv = unetConv2(in_size, out_size, is_batchnorm)  # modified following unetup3

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super().__init__()
        self.conv = unetConv2(in_size + out_size, out_size, is_batchnorm, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super().__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='bilinear'), )

    def forward(self, x):
        return self.dsv(x)


class UnetConv3(nn.Module):
    def __init__(
            self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
            init_stride=(1, 1, 1)
    ):
        super().__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3_CT(nn.Module):
    def __init__(
            self, in_size, out_size, is_batchnorm=True, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
            scale_factor=(2, 2, 2)
    ):
        super().__init__()
        self.conv = UnetConv3(
            in_size + out_size, out_size, is_batchnorm, kernel_size=kernel_size, padding_size=padding_size)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
