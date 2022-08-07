# -*- coding: utf-8 -*-
""" nns/models/unet/models/unet3d """

import torch.nn.functional as F
from torch import nn

from nns.models.unet.network_others import init_weights
from nns.models.unet.utils import UnetConv3, UnetUp_CT


__all__ = ['UNet3D']


class UNet3D(nn.Module):
    """
    Original 3D UNet from Attention Gated Networks
    source: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_3D.py
    """

    def __init__(self, feature_scale=4, n_classes=21, n_channels=3, is_batchnorm=True):
        super().__init__()
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        maxpool_kernel_size = (2, 2, 2)

        # downsampling
        self.conv1 = UnetConv3(self.n_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=maxpool_kernel_size)

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=maxpool_kernel_size)

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=maxpool_kernel_size)

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=maxpool_kernel_size)

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp_CT(filters[4], filters[3], is_batchnorm, data_dimensions=3,
                                    scale_factor=maxpool_kernel_size)
        self.up_concat3 = UnetUp_CT(filters[3], filters[2], is_batchnorm, data_dimensions=3,
                                    scale_factor=maxpool_kernel_size)
        self.up_concat2 = UnetUp_CT(filters[2], filters[1], is_batchnorm, data_dimensions=3,
                                    scale_factor=maxpool_kernel_size)
        self.up_concat1 = UnetUp_CT(filters[1], filters[0], is_batchnorm, data_dimensions=3,
                                    scale_factor=maxpool_kernel_size)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], self.n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
