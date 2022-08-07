# -*- coding: utf-8 -*-
"""
nns/models/unet/utils

Source: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/utils.py
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from nns.models.unet.network_others import init_weights


__all__ = [
    'unetConvX', 'UnetGridGatingSignal',  'unetUp', 'UnetUp_CT', 'UnetDsv', 'UnetConv3',
]


class unetConvX(nn.Module):
    def __init__(
            self, in_size: int, out_size: int, is_batchnorm: bool = True, n: int = 2,
            kernel_size: Union[int, tuple] = 3, stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 1,  data_dimensions: int = 2,
            batchnorm_cls: Optional[_BatchNorm] = None):
        """
        kwargs:
            in_size          <int>: input channels
            out_size         <int>: output channels
            is_batchnorm    <bool>: Whether or not use batch normalization. Default True
            n                <int>: Number of conv + bn? + relu blocks to stack. Default 2
            kernel_size <int|tuple>: Default 3
            stride     <int|tuple>: Default 1
            padding    <int|tuple>: Default 1
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [batch, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
        """
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.is_batchnorm = is_batchnorm
        self.n = n
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_dimensions = data_dimensions
        self.batchnorm_cls = batchnorm_cls

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_size, int), type(self.in_size)
        assert self.in_size > 0, self.in_size
        assert isinstance(self.out_size, int), type(self.out_size)
        assert self.out_size > 0, self.out_size
        assert isinstance(self.is_batchnorm, bool), type(self.is_batchnorm)
        assert isinstance(self.n, int), type(self.n)
        assert self.n > 0, self.n
        assert isinstance(self.kernel_size, (int, tuple)), type(self.kernel_size)
        assert isinstance(self.stride, (int, tuple)), type(self.stride)
        assert isinstance(self.padding, (int, tuple)), type(self.padding)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)

        convxd = nn.Conv2d if data_dimensions == 2 else nn.Conv3d
        batchnorm = nn.BatchNorm2d if data_dimensions == 2 else nn.BatchNorm3d

        in_channels = self.in_size
        out_channels = self.out_size

        if self.is_batchnorm:
            for i in range(1, self.n+1):
                conv = nn.Sequential(
                    convxd(in_channels, out_channels, self.kernel_size, self.stride, self.padding),
                    batchnorm(out_channels),
                    nn.ReLU(inplace=True),
                )
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels
        else:
            for i in range(1, self.n+1):
                conv = nn.Sequential(
                    convxd(in_channels, out_channels, self.kernel_size, self.stride, self.padding),
                    nn.ReLU(inplace=True),
                )
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels

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
    def __init__(self, in_size, out_size, kernel_size=1, is_batchnorm=True, data_dimensions: int = 2):
        super().__init__()

        assert data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = nn.Conv2d if data_dimensions == 2 else nn.Conv3d
        batchnorm = nn.BatchNorm2d if data_dimensions == 2 else nn.BatchNorm3d

        if is_batchnorm:
            self.conv1 = nn.Sequential(convxd(in_size, out_size, kernel_size, stride=1, padding=0),
                                       batchnorm(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(convxd(in_size, out_size, kernel_size, stride=1, padding=0),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size: int, out_size: int, is_deconv: bool, is_batchnorm: bool = True,
                 data_dimensions: int = 2):
        super().__init__()

        assert data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        if is_deconv:
            # self.conv = unetConv2(in_size, out_size, False)  # original
            self.conv = unetConvX(in_size, out_size, is_batchnorm,
                                  data_dimensions=data_dimensions)  # modified following unetup3
            convtransposexd = nn.ConvTranspose2d if data_dimensions == 2 else nn.ConvTranspose3d
            self.up = convtransposexd(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConvX(in_size+out_size, out_size, is_batchnorm,
                                  data_dimensions=data_dimensions)  # modified following unetup3
            if data_dimensions == 2:
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            else:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConvX') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp_CT(nn.Module):
    def __init__(
            self, in_size: int, out_size: int, is_batchnorm: bool = True, data_dimensions: int = 2,
            kernel_size: Union[int, tuple] = 3, padding_size: Union[int, tuple] = 1,
            scale_factor: Union[int, tuple] = 2
    ):
        super().__init__()

        assert data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(scale_factor, (int, tuple)), type(scale_factor)

        self.conv = unetConvX(
            in_size + out_size, out_size, is_batchnorm, kernel_size=kernel_size, padding=padding_size,
            data_dimensions=data_dimensions
        )
        mode = 'bilinear' if data_dimensions == 2 else 'trilinear'
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConvX') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor, data_dimensions: int = 2):
        super().__init__()

        assert data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = nn.Conv2d if data_dimensions == 2 else nn.Conv3d
        mode = 'bilinear' if data_dimensions == 2 else 'trilinear'
        self.dsv = nn.Sequential(convxd(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode=mode))

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
