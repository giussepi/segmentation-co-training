# -*- coding: utf-8 -*-
""" nns/models/unet_3plus_da """

from typing import Optional

import numpy as np
import torch
from gtorch_utils.nns.models.segmentation import UNet_3Plus
from gtorch_utils.utils.images import apply_padding

from nns.models.da_model import BaseDATrain
from nns.models.layers.disagreement_attention import ThresholdedDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.layers import DAConvBlock


__all__ = ['UNet_3Plus_DA', 'UNet_3Plus_DA_Train']


class UNet_3Plus_DA(UNet_3Plus):
    """
    UNet_3Plus with disagreement attention between activations maps with the same dimensions
    Note: DA blocks added only to the encoder
    """

    def __init__(self, da_threshold: float = np.inf,
                 da_block_cls: BaseDisagreementAttentionBlock = ThresholdedDisagreementAttentionBlock,
                 da_block_config: Optional[dict] = None, **kwargs):
        """
        Kwargs:
            da_threshold   <float>: threshold to apply attention or not. Only when
                                    metric2 > da_threshold the attention is applied.
                                    If you want to always apply disagreement attention set
                                    it to np.NINF.
                                    Default np.inf (no disaggrement attention applied)
            da_block_cls <BaseDisagreementAttentionBlock>: class descendant of BaseDisagreementAttentionBlock
                                    Default ThresholdedDisagreementAttentionBlock
            da_block_config <dict>: Configuration for disagreement attention block.
                                    Default None
        """
        super().__init__(**kwargs)
        assert isinstance(da_threshold, float), type(da_threshold)
        assert issubclass(da_block_cls, BaseDisagreementAttentionBlock), \
            f'{da_block_cls} is not a descendant of BaseDisagreementAttentionBlock'
        if da_block_config:
            assert isinstance(da_block_config, dict), type(da_block_config)
            self.da_block_config = da_block_config
        else:
            self.da_block_config = {}

        self.filters = [64, 128, 256, 512, 1024]

        self.da_threshold = da_threshold
        # disagreement attention between conv1 layers
        self.da_conv1 = DAConvBlock(
            da_block_cls(self.filters[0], self.filters[0], **self.da_block_config), 2*self.filters[0],
            self.filters[0])
        # disagreement attention between conv2 layers
        self.da_conv2 = DAConvBlock(
            da_block_cls(self.filters[1], self.filters[1], **self.da_block_config), 2*self.filters[1],
            self.filters[1])
        # disagreement attention between conv3 layers
        self.da_conv3 = DAConvBlock(
            da_block_cls(self.filters[2], self.filters[2], **self.da_block_config), 2*self.filters[2],
            self.filters[2])
        # disagreement attention between conv4 layers
        self.da_conv4 = DAConvBlock(
            da_block_cls(self.filters[3], self.filters[3], **self.da_block_config), 2*self.filters[3],
            self.filters[3])
        # disagreement attention between conv5 layers
        self.da_conv5 = DAConvBlock(
            da_block_cls(self.filters[4], self.filters[4], **self.da_block_config), 2*self.filters[4],
            self.filters[4])

    def forward_1(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        h1 = self.conv1(x)  # h1->320*320*64

        return {'h1': h1}

    def forward_2(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        x['h1'] = self.da_conv1(x['h1'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        x['h2'] = self.maxpool1(x['h1'])
        x['h2'] = self.conv2(x['h2'])  # h2->160*160*128

        return x

    def forward_3(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        x['h2'] = self.da_conv2(x['h2'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        x['h3'] = self.maxpool2(x['h2'])
        x['h3'] = self.conv3(x['h3'])  # h3->80*80*256

        return x

    def forward_4(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        x['h3'] = self.da_conv3(x['h3'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        x['h4'] = self.maxpool3(x['h3'])
        x['h4'] = self.conv4(x['h4'])  # h4->40*40*512

        return x

    def forward_5(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        x['h4'] = self.da_conv4(x['h4'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        x['h5'] = self.maxpool4(x['h4'])
        x['hd5'] = self.conv5(x['h5'])  # h5->20*20*1024

        return x

    def forward_6(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        x['hd5'] = self.da_conv5(x['hd5'], skip_connection, disable_attention=metric2 <= self.da_threshold)

        return x

    def forward_7(self, x: dict):
        assert isinstance(x, dict), type(x)

        # -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(x['h1']))))
        h1_PT_hd4 = apply_padding(h1_PT_hd4, x['h4'])
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(x['h2']))))
        h2_PT_hd4 = apply_padding(h2_PT_hd4, x['h4'])
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(x['h3']))))
        h3_PT_hd4 = apply_padding(h3_PT_hd4, x['h4'])
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(x['h4'])))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(x['hd5']))))
        hd5_UT_hd4 = apply_padding(hd5_UT_hd4, x['h4'])
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(x['h1']))))
        h1_PT_hd3 = apply_padding(h1_PT_hd3, x['h3'])
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(x['h2']))))
        h2_PT_hd3 = apply_padding(h2_PT_hd3, x['h3'])
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(x['h3'])))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd4_UT_hd3 = apply_padding(hd4_UT_hd3, x['h3'])
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(x['hd5']))))
        hd5_UT_hd3 = apply_padding(hd5_UT_hd3, x['h3'])
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(x['h1']))))
        h1_PT_hd2 = apply_padding(h1_PT_hd2, x['h2'])
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(x['h2'])))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd3_UT_hd2 = apply_padding(hd3_UT_hd2, x['h2'])
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd4_UT_hd2 = apply_padding(hd4_UT_hd2, x['h2'])
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(x['hd5']))))
        hd5_UT_hd2 = apply_padding(hd5_UT_hd2, x['h2'])
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(x['h1'])))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd2_UT_hd1 = apply_padding(hd2_UT_hd1, x['h1'])
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd3_UT_hd1 = apply_padding(hd3_UT_hd1, x['h1'])
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd4_UT_hd1 = apply_padding(hd4_UT_hd1, x['h1'])
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(x['hd5']))))
        hd5_UT_hd1 = apply_padding(hd5_UT_hd1, x['h1'])
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes

        return d1

    def forward(self, x: torch.Tensor):
        """ forward pass without disagreement attention (called when working with a single model) """
        assert isinstance(x, torch.Tensor), type(x)

        x = self.forward_1(x)
        x = self.forward_2(x, x['h1'], np.inf)
        x = self.forward_3(x, x['h2'], np.inf)
        x = self.forward_4(x, x['h3'], np.inf)
        x = self.forward_5(x, x['h4'], np.inf)
        x = self.forward_6(x, x['hd5'], np.inf)
        x = self.forward_7(x)

        return x


class UNet_3Plus_DA_Train(BaseDATrain):
    """
    Disagreement attention trainer class for two UNet_3Plus

    Usage:
        mymodel = UNet_3Plus_DA_Train(
            model1_cls=model1_cls, kwargs1=kwargs1, model1_cls=model2_cls, kwargs2=kwargs2)
    """

    def forward(self, x: torch.Tensor,  metric1: float, metric2: float):
        """
        Forward pass with disagreement attention (called during training)

        Kwargs:
            x <torch.Tensor>: input
            metric1  <float>: metric from model 1 to be compared with da_threshold to activate
                              or not the disagreement attention
            metric2  <float>: metric from model 2 to be compared with da_threshold to activate
                              or not the disagreement attention

        returns:
            logits_model1 <torch.Tensor>, logits_model2 <torch.Tensor>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(metric1, float), type(metric1)
        assert isinstance(metric2, float), type(metric2)

        x1 = self.model1.forward_1(x)
        x2 = self.model2.forward_1(x)
        # FIXME: I don't think using x1_ will keep the previous values from x1_
        # try x1, x2 = self.model1.forward_2(x1, x2['h1'], metric2), self.model2.forward_2(x2, x1['h1'], metric1)
        x1_ = self.model1.forward_2(x1, x2['h1'], metric2)
        x2 = self.model2.forward_2(x2, x1['h1'], metric1)
        x1 = self.model1.forward_3(x1_, x2['h2'], metric2)
        x2 = self.model2.forward_3(x2, x1_['h2'], metric1)
        x1_ = self.model1.forward_4(x1, x2['h3'], metric2)
        x2 = self.model2.forward_4(x2, x1['h3'], metric1)
        x1 = self.model1.forward_5(x1_, x2['h4'], metric2)
        x2 = self.model2.forward_5(x2, x1_['h4'], metric1)
        x1_ = self.model1.forward_6(x1, x2['hd5'], metric2)
        x2 = self.model2.forward_6(x2, x1['hd5'], metric1)
        x1 = self.model1.forward_7(x1_)
        x2 = self.model2.forward_7(x2)

        return x1, x2
