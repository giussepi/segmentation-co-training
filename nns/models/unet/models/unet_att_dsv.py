# -*- coding: utf-8 -*-
""" nns/models/unet/models/unet_att_dsv """

from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn

from nns.models.unet.layers import GridAttentionBlockxD
from nns.models.unet.utils import unetConvX, UnetUp_CT, UnetGridGatingSignal, UnetDsv
from nns.models.unet.network_others import init_weights


__all__ = ['SingleAttentionBlock', 'MultiAttentionBlock', 'UNet_Att_DSV']


class SingleAttentionBlock(nn.Module):
    """

    Source: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_CT_single_att_dsv_3D.py#L113
    """

    def __init__(
            self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor, data_dimensions: int = 2):
        super().__init__()
        assert data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        self.gate_block_1 = GridAttentionBlockxD(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor,
                                                 data_dimensions=data_dimensions
                                                 )

        convxd = nn.Conv2d if data_dimensions == 2 else nn.Conv3d
        batchnomxd = nn.BatchNorm2d if data_dimensions == 2 else nn.BatchNorm3d

        self.combine_gates = nn.Sequential(convxd(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           batchnomxd(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, x, gating_signal):
        gate_1, attention_1 = self.gate_block_1(x, gating_signal)

        return self.combine_gates(gate_1), attention_1


class MultiAttentionBlock(nn.Module):
    """

    Source: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_CT_multi_att_dsv_3D.py#L113
    """

    def __init__(
            self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor, data_dimensions: int = 2):
        super().__init__()

        assert data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        self.gate_block_1 = GridAttentionBlockxD(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor,
                                                 data_dimensions=data_dimensions
                                                 )
        self.gate_block_2 = GridAttentionBlockxD(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor,
                                                 data_dimensions=data_dimensions
                                                 )

        convxd = nn.Conv2d if data_dimensions == 2 else nn.Conv3d
        batchnomxd = nn.BatchNorm2d if data_dimensions == 2 else nn.BatchNorm3d

        self.combine_gates = nn.Sequential(
            convxd(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
            batchnomxd(in_size),
            nn.ReLU(inplace=True)
        )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, x, gating_signal):
        gate_1, attention_1 = self.gate_block_1(x, gating_signal)
        gate_2, attention_2 = self.gate_block_2(x, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


class UNet_Att_DSV(nn.Module):
    """

    Based on https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_CT_single_att_dsv_3D.py
    """

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, n_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=2, is_batchnorm=True,
                 attention_block_cls=SingleAttentionBlock, data_dimensions: int = 2):
        super().__init__()
        self.is_deconv = is_deconv
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.attention_block_cls = attention_block_cls
        self.data_dimensions = data_dimensions

        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        maxpool = nn.MaxPool2d if self.data_dimensions == 2 else nn.MaxPool3d
        self.conv1 = unetConvX(self.n_channels, filters[0], self.is_batchnorm,
                               kernel_size=3, padding=1, data_dimensions=self.data_dimensions)
        self.maxpool1 = maxpool(kernel_size=2)

        self.conv2 = unetConvX(filters[0], filters[1], self.is_batchnorm, kernel_size=3, padding=1,
                               data_dimensions=self.data_dimensions)
        self.maxpool2 = maxpool(kernel_size=2)

        self.conv3 = unetConvX(filters[1], filters[2], self.is_batchnorm, kernel_size=3, padding=1,
                               data_dimensions=self.data_dimensions)
        self.maxpool3 = maxpool(kernel_size=2)

        self.conv4 = unetConvX(filters[2], filters[3], self.is_batchnorm, kernel_size=3, padding=1,
                               data_dimensions=self.data_dimensions)
        self.maxpool4 = maxpool(kernel_size=2)

        self.center = unetConvX(filters[3], filters[4], self.is_batchnorm, kernel_size=3, padding=1,
                                data_dimensions=self.data_dimensions)
        self.gating = UnetGridGatingSignal(
            filters[4], filters[4], kernel_size=1, is_batchnorm=self.is_batchnorm,
            data_dimensions=self.data_dimensions)

        # attention blocks
        self.attentionblock2 = self.attention_block_cls(
            in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample, data_dimensions=self.data_dimensions)
        self.attentionblock3 = self.attention_block_cls(
            in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample, data_dimensions=self.data_dimensions)
        self.attentionblock4 = self.attention_block_cls(
            in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
            nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample, data_dimensions=self.data_dimensions)

        # upsampling
        self.up_concat4 = UnetUp_CT(filters[4], filters[3], is_batchnorm, data_dimensions=self.data_dimensions)
        self.up_concat3 = UnetUp_CT(filters[3], filters[2], is_batchnorm, data_dimensions=self.data_dimensions)
        self.up_concat2 = UnetUp_CT(filters[2], filters[1], is_batchnorm, data_dimensions=self.data_dimensions)
        self.up_concat1 = UnetUp_CT(filters[1], filters[0], is_batchnorm, data_dimensions=self.data_dimensions)

        # deep supervision
        self.dsv4 = UnetDsv(in_size=filters[3], out_size=n_classes, scale_factor=8,
                            data_dimensions=self.data_dimensions)  # tweak 2: scale=2
        self.dsv3 = UnetDsv(in_size=filters[2], out_size=n_classes, scale_factor=4,
                            data_dimensions=self.data_dimensions)  # tweak 2: scale=2
        self.dsv2 = UnetDsv(in_size=filters[1], out_size=n_classes, scale_factor=2,
                            data_dimensions=self.data_dimensions)

        convxd = nn.Conv2d if self.data_dimensions == 2 else nn.Conv3d
        self.dsv1 = convxd(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = convxd(n_classes*4, n_classes, 1)  # original
        # self.final = nn.Conv2d(n_classes, n_classes, 1)   # tweak for 1 & 2

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
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))  # original
        # final = self.final(reduce(torch.add, [dsv4, dsv3, dsv2, dsv1]))  # tweak 1
        # tweak 2
        # dsv3 = self.dsv4(up4) + up3
        # dsv2 = self.dsv3(dsv3) + up2
        # dsv1 = self.dsv2(dsv2) + up1
        # dsv = self.dsv1(dsv1)
        # final = self.final(dsv)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
