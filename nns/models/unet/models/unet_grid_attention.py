# -*- coding: utf-8 -*-
""" nns/models/unet/models/unet_grind_attention """

from torch import nn
import torch.nn.functional as F

from nns.models.unet.layers import GridAttentionBlockxD
from nns.models.unet.network_others import init_weights
from nns.models.unet.utils import unetConvX, unetUp, UnetGridGatingSignal


__all__ = ['UNet_Grid_Attention']


class UNet_Grid_Attention(nn.Module):
    """

    Source: Based on https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_grid_attention_3D.py
    """

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, n_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=2, is_batchnorm=True,
                 data_dimensions: int = 2):
        super().__init__()
        self.is_deconv = is_deconv
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.data_dimensions = data_dimensions

        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        maxpool = nn.MaxPool2d if self.data_dimensions == 2 else nn.MaxPool3d
        self.conv1 = unetConvX(self.n_channels, filters[0], self.is_batchnorm,
                               data_dimensions=self.data_dimensions)
        self.maxpool1 = maxpool(kernel_size=2)

        self.conv2 = unetConvX(filters[0], filters[1], self.is_batchnorm,
                               data_dimensions=self.data_dimensions)
        self.maxpool2 = maxpool(kernel_size=2)

        self.conv3 = unetConvX(filters[1], filters[2], self.is_batchnorm,
                               data_dimensions=self.data_dimensions)
        self.maxpool3 = maxpool(kernel_size=2)

        self.conv4 = unetConvX(filters[2], filters[3], self.is_batchnorm,
                               data_dimensions=self.data_dimensions)
        self.maxpool4 = maxpool(kernel_size=2)

        self.center = unetConvX(filters[3], filters[4], self.is_batchnorm,
                                data_dimensions=self.data_dimensions)
        self.gating = UnetGridGatingSignal(
            filters[4], filters[3], kernel_size=1, is_batchnorm=self.is_batchnorm,
            data_dimensions=self.data_dimensions)

        # attention blocks
        self.attentionblock2 = GridAttentionBlockxD(
            in_channels=filters[1], gating_channels=filters[3],
            inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=nonlocal_mode,
            data_dimensions=self.data_dimensions)
        self.attentionblock3 = GridAttentionBlockxD(
            in_channels=filters[2], gating_channels=filters[3],
            inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=nonlocal_mode,
            data_dimensions=self.data_dimensions)
        self.attentionblock4 = GridAttentionBlockxD(
            in_channels=filters[3], gating_channels=filters[3],
            inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=nonlocal_mode,
            data_dimensions=self.data_dimensions)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm,
                                 data_dimensions=self.data_dimensions)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm,
                                 data_dimensions=self.data_dimensions)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm,
                                 data_dimensions=self.data_dimensions)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm,
                                 data_dimensions=self.data_dimensions)

        # final conv (without any concat)
        convxd = nn.Conv2d if self.data_dimensions == 2 else nn.Conv3d
        self.final = convxd(filters[0], n_classes, 1)

        batchnorm = nn.BatchNorm2d if self.data_dimensions == 2 else nn.BatchNorm3d

        # initialise weights
        for m in self.modules():
            if isinstance(m, convxd):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, batchnorm):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        g_conv2, att2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
