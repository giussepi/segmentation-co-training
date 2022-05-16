# -*- coding: utf-8 -*-
""" nns/models/unet_3plus_da2ext """
from typing import Optional

import numpy as np
import torch
from gtorch_utils.utils.images import apply_padding

from nns.models.da_model import BaseDATrain
from nns.models.layers.disagreement_attention import ThresholdedDisagreementAttentionBlock, \
    AttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.constants import AttentionMergingType
from nns.models.layers.disagreement_attention.layers import DAConvBlock, AttentionMergingBlock
from nns.models import UNet_3Plus_DA


__all__ = ['UNet_3Plus_DA2Ext', 'UNet_3Plus_DA2Ext_Train']


class UNet_3Plus_DA2Ext(UNet_3Plus_DA):
    """
    UNet_3Plus with disagreement attention between activations maps with the same dimensions and also
    gating signals in the decoders.
    Note: DA blocks through all the encoder and decoder
    """

    def __init__(self,
                 da_threshold: float = np.inf,
                 da_block_cls: BaseDisagreementAttentionBlock = ThresholdedDisagreementAttentionBlock,
                 da_block_config: Optional[dict] = None, da_merging_type: str = AttentionMergingType.SUM,
                 **kwargs):
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
            da_merging_type  <str>: Valid attention merging strategy
                                    (see nns.models.layers.disagreement_attention.constants.AttentionMergingType).
                                    Default AttentionMergingType.SUM
        """
        super().__init__(da_threshold, da_block_cls, da_block_config, **kwargs)

        # attention merging infrastructure for hd4  ###########################
        # disagreement attention after relu4d_1 (hd4)
        self.da_hd4 = DAConvBlock(
            da_block_cls(self.UpChannels, self.UpChannels, **self.da_block_config),
            2*self.UpChannels, self.UpChannels, only_attention=True
        )
        # gating signal for hd4
        self.gsa_hd5_to_hd4 = AttentionBlock(
            self.UpChannels, self.filters[4], resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        # attention merging block for hd4
        self.att_merging_block_for_hd4 = AttentionMergingBlock(
            2*self.UpChannels, self.UpChannels, merging_type=da_merging_type)

        # attention merging infrastructure for hd3  ###########################
        # disagreement attention after relu3d_1 (hd3)
        self.da_hd3 = DAConvBlock(
            da_block_cls(self.UpChannels, self.UpChannels, **self.da_block_config),
            2*self.UpChannels, self.UpChannels, only_attention=True
        )
        # gating signal for hd3
        self.gsa_hd4_to_hd3 = AttentionBlock(
            self.UpChannels, self.UpChannels, resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        # attention merging block for hd3
        self.att_merging_block_for_hd3 = AttentionMergingBlock(
            2*self.UpChannels, self.UpChannels, merging_type=da_merging_type)

        # attention merging infrastructure for hd2  ###########################
        # disagreement attention after relu2d_1 (hd2)
        self.da_hd2 = DAConvBlock(
            da_block_cls(self.UpChannels, self.UpChannels, **self.da_block_config),
            2*self.UpChannels, self.UpChannels, only_attention=True
        )
        # gating signal for hd2
        self.gsa_hd3_to_hd2 = AttentionBlock(
            self.UpChannels, self.UpChannels, resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        # attention merging block for hd2
        self.att_merging_block_for_hd2 = AttentionMergingBlock(
            2*self.UpChannels, self.UpChannels, merging_type=da_merging_type)

        # attention merging infrastructure for hd1  ###########################
        # disagreement attention after relu1d_1 (hd1)
        # FIXME: not sure this attention block is necessary maybe not....
        # self.da_hd1 = DAConvBlock(
        #     da_block_cls(self.UpChannels, self.UpChannels, **self.da_block_config),
        #     2*self.UpChannels, self.UpChannels, only_attention=True
        # )
        # gating signal for hd1
        # self.gsa_hd2_to_hd1 = GatingSignalAttentionBlock(
        #     self.UpChannels, self.UpChannels, resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        # attention merging block for hd1
        # self.att_merging_block_for_hd1 = AttentionMergingBlock(
        #     2*self.UpChannels, self.UpChannels, merging_type=da_merging_type)

    def forward_7(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_PT_hd4'] = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(x['h1A']))))
        x['h1_PT_hd4'] = apply_padding(x['h1_PT_hd4'], x['h4A'])
        x['h2_PT_hd4'] = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(x['h2A']))))
        x['h2_PT_hd4'] = apply_padding(x['h2_PT_hd4'], x['h4A'])
        x['h3_PT_hd4'] = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(x['h3A']))))
        x['h3_PT_hd4'] = apply_padding(x['h3_PT_hd4'], x['h4A'])
        x['h4_Cat_hd4'] = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(x['h4A'])))
        x['hd5_UT_hd4'] = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(
            self.hd5_UT_hd4(x['hd5A']))))
        x['hd5_UT_hd4'] = apply_padding(x['hd5_UT_hd4'], x['h4A'])
        x['hd4'] = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((x['h1_PT_hd4'], x['h2_PT_hd4'], x['h3_PT_hd4'], x['h4_Cat_hd4'], x['hd5_UT_hd4']), 1)
        )))  # hd4->40*40*UpChannels

        return x

    def forward_8(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        da_hd4 = self.da_hd4(x['hd4'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        _, gsa_hd5_to_hd4 = self.gsa_hd5_to_hd4(x['hd4'], x['hd5A'])
        x['hd4A'] = self.att_merging_block_for_hd4(
            x['hd4'], gsa_hd5_to_hd4, da_hd4, disable_da=metric2 < self.da_threshold)
        x['h1_PT_hd3'] = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(x['h1A']))))
        x['h1_PT_hd3'] = apply_padding(x['h1_PT_hd3'], x['h3A'])
        x['h2_PT_hd3'] = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(x['h2A']))))
        x['h2_PT_hd3'] = apply_padding(x['h2_PT_hd3'], x['h3A'])
        x['h3_Cat_hd3'] = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(x['h3A'])))
        x['hd4_UT_hd3'] = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(
            self.hd4_UT_hd3(x['hd4A']))))
        x['hd4_UT_hd3'] = apply_padding(x['hd4_UT_hd3'], x['h3A'])
        x['hd5_UT_hd3'] = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(
            self.hd5_UT_hd3(x['hd5A']))))
        x['hd5_UT_hd3'] = apply_padding(x['hd5_UT_hd3'], x['h3A'])
        x['hd3'] = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((x['h1_PT_hd3'], x['h2_PT_hd3'], x['h3_Cat_hd3'], x['hd4_UT_hd3'], x['hd5_UT_hd3']), 1)
        )))  # hd3->80*80*UpChannels

        return x

    def forward_9(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        da_hd3 = self.da_hd3(x['hd3'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        _, gsa_hd4_to_hd3 = self.gsa_hd4_to_hd3(x['hd3'], x['hd4A'])
        x['hd3A'] = self.att_merging_block_for_hd3(
            x['hd3'], gsa_hd4_to_hd3, da_hd3, disable_da=metric2 <= self.da_threshold)
        x['h1_PT_hd2'] = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(x['h1A']))))
        x['h1_PT_hd2'] = apply_padding(x['h1_PT_hd2'], x['h2A'])
        x['h2_Cat_hd2'] = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(x['h2A'])))
        x['hd3_UT_hd2'] = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(
            self.hd3_UT_hd2(x['hd3A']))))
        x['hd3_UT_hd2'] = apply_padding(x['hd3_UT_hd2'], x['h2A'])
        x['hd4_UT_hd2'] = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(
            self.hd4_UT_hd2(x['hd4A']))))
        x['hd4_UT_hd2'] = apply_padding(x['hd4_UT_hd2'], x['h2A'])
        x['hd5_UT_hd2'] = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(
            self.hd5_UT_hd2(x['hd5A']))))
        x['hd5_UT_hd2'] = apply_padding(x['hd5_UT_hd2'], x['h2A'])
        x['hd2'] = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((x['h1_PT_hd2'], x['h2_Cat_hd2'], x['hd3_UT_hd2'], x['hd4_UT_hd2'], x['hd5_UT_hd2']), 1)
        )))  # hd2->160*160*UpChannels

        return x

    def forward_10(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        da_hd2 = self.da_hd2(x['hd2'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        _, gsa_hd3_to_hd2 = self.gsa_hd3_to_hd2(x['hd2'], x['hd3A'])
        x['hd2A'] = self.att_merging_block_for_hd2(
            x['hd2'], gsa_hd3_to_hd2, da_hd2, disable_da=metric2 <= self.da_threshold)
        x['h1_Cat_hd1'] = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(x['h1A'])))
        x['hd2_UT_hd1'] = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(
            self.hd2_UT_hd1(x['hd2A']))))
        x['hd2_UT_hd1'] = apply_padding(x['hd2_UT_hd1'], x['h1A'])
        x['hd3_UT_hd1'] = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(
            self.hd3_UT_hd1(x['hd3A']))))
        x['hd3_UT_hd1'] = apply_padding(x['hd3_UT_hd1'], x['h1A'])
        x['hd4_UT_hd1'] = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(
            self.hd4_UT_hd1(x['hd4A']))))
        x['hd4_UT_hd1'] = apply_padding(x['hd4_UT_hd1'], x['h1A'])
        x['hd5_UT_hd1'] = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(
            self.hd5_UT_hd1(x['hd5A']))))
        x['hd5_UT_hd1'] = apply_padding(x['hd5_UT_hd1'], x['h1A'])
        x['hd1'] = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((x['h1_Cat_hd1'], x['hd2_UT_hd1'], x['hd3_UT_hd1'], x['hd4_UT_hd1'], x['hd5_UT_hd1']), 1)
        )))  # hd1->320*320*UpChannels

        return x

    def forward_11(self, x: dict, skip_connection: torch.Tensor, metric2: float):
        assert isinstance(x, dict), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(metric2, float), type(metric2)

        # da_hd1 = self.da_hd1(x['hd1'], skip_connection, disable_attention=metric2 <= self.da_threshold)
        # _, gsa_hd2_to_hd1 = self.gsa_hd2_hd1(x['hd1'], x['hd2A'])
        # x['hd1A'] = self.att_merging_block_for_hd1(
        #     x['hd1'], gsa_hd2_to_hd1, da_hd1, disable_da=metric2 <= self.da_threshold)

        return x

    def forward_12(self, x: dict):
        assert isinstance(x, dict), type(x)

        # d1 = self.outconv1(x['hd1A'])  # d1->320*320*n_classes
        d1 = self.outconv1(x['hd1'])  # d1->320*320*n_classes

        return d1

    def forward(self, x: torch.Tensor):
        """ forward pass without disagreement attention (called when working with a single model) """
        assert isinstance(x, torch.Tensor), type(x)

        #######################################################################
        #                               encoder                               #
        #######################################################################
        x = self.forward_1(x)
        x = self.forward_2(x, x['h1'], np.inf)
        x = self.forward_3(x, x['h2'], np.inf)
        x = self.forward_4(x, x['h3'], np.inf)
        x = self.forward_5(x, x['h4'], np.inf)
        x = self.forward_6(x, x['hd5'], np.inf)
        #######################################################################
        #                               decoder                               #
        #######################################################################
        x = self.forward_7(x)
        x = self.forward_8(x, x['hd4'], np.inf)
        x = self.forward_9(x, x['hd3'], np.inf)
        x = self.forward_10(x, x['hd2'], np.inf)
        x = self.forward_11(x, x['hd1'], np.inf)
        x = self.forward_12(x)

        return x


class UNet_3Plus_DA2Ext_Train(BaseDATrain):
    """
    Disagreement attention trainer class for two UNet_3Plus

    Usage:
        mymodel = UNet_3Plus_DA_Train2(
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

        #######################################################################
        #                               encoder                               #
        #######################################################################
        x1 = self.model1.forward_1(x)
        x2 = self.model2.forward_1(x)
        x1 = self.model1.forward_2(x1, x2['h1'], metric2)
        x2 = self.model2.forward_2(x2, x1['h1'], metric1)
        x1 = self.model1.forward_3(x1, x2['h2'], metric2)
        x2 = self.model2.forward_3(x2, x1['h2'], metric1)
        x1 = self.model1.forward_4(x1, x2['h3'], metric2)
        x2 = self.model2.forward_4(x2, x1['h3'], metric1)
        x1 = self.model1.forward_5(x1, x2['h4'], metric2)
        x2 = self.model2.forward_5(x2, x1['h4'], metric1)
        x1 = self.model1.forward_6(x1, x2['hd5'], metric2)
        x2 = self.model2.forward_6(x2, x1['hd5'], metric1)
        #######################################################################
        #                               decoder                               #
        #######################################################################
        x1 = self.model1.forward_7(x1)
        x2 = self.model2.forward_7(x2)
        x1 = self.model1.forward_8(x1, x2['hd4'], metric2)
        x2 = self.model2.forward_8(x2, x1['hd4'], metric1)
        x1 = self.model1.forward_9(x1, x2['hd3'], metric2)
        x2 = self.model2.forward_9(x2, x1['hd3'], metric1)
        x1 = self.model1.forward_10(x1, x2['hd2'], metric2)
        x2 = self.model2.forward_10(x2, x1['hd2'], metric1)
        x1 = self.model1.forward_11(x1, x2['hd1'], metric2)
        x2 = self.model2.forward_11(x2, x1['hd1'], metric1)
        x1 = self.model1.forward_12(x1)
        x2 = self.model2.forward_12(x2)

        return x1, x2
