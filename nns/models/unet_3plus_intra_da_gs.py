# -*- coding: utf-8 -*-
""" nns/models/unet_3plus_intra_da_gs """

from typing import Optional

import torch
from gtorch_utils.nns.models.segmentation import UNet_3Plus
from gtorch_utils.utils.images import apply_padding

from nns.models.layers.disagreement_attention.inter_class import ThresholdedDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.layers import DAConvBlockGS
from nns.models.mixins import InitMixin


__all__ = ['UNet_3Plus_Intra_DA_GS']


class UNet_3Plus_Intra_DA_GS(UNet_3Plus, InitMixin):
    """
    UNet_3Plus with intra-model disagreement attention using gating signal

    Note: batchnorm_cls and init_type kwargs are copied into da_block_config kwarg
    """

    def __init__(self, da_block_cls: BaseDisagreementAttentionBlock = ThresholdedDisagreementAttentionBlock,
                 da_block_config: Optional[dict] = None, **kwargs):
        """
        Kwargs:
            da_block_cls <BaseDisagreementAttentionBlock>: class descendant of BaseDisagreementAttentionBlock
                                    Default ThresholdedDisagreementAttentionBlock
            da_block_config <dict>: Configuration for disagreement attention block.
                                    Default None
        """
        super().__init__(**kwargs)
        assert issubclass(da_block_cls, BaseDisagreementAttentionBlock), \
            f'{da_block_cls} is not a descendant of BaseDisagreementAttentionBlock'
        if da_block_config:
            assert isinstance(da_block_config, dict), type(da_block_config)
            self.da_block_config = da_block_config
        else:
            self.da_block_config = {}
        # adding extra configuration to da_block_config
        self.da_block_config['batchnorm_cls'] = kwargs.get('batchnorm_cls')
        self.da_block_config['init_type'] = kwargs.get('init_type')

        self.filters = [64, 128, 256, 512, 1024]

        # intra-class DA ga-h4 & skip-con-hd5 -> hd4
        self.intra_da_hd4 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[3], self.filters[4], resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[3] + self.UpChannels,
            self.UpChannels
        )
        # TODO: modify self.conv4d_1 to process the intra_da_hd4, This could improve the results
        # intra-class DA ga-h3 & skip-con-hd4 -> hd3
        self.intra_da_hd3 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[2], self.UpChannels, resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[2] + self.UpChannels,
            self.UpChannels
        )
        # TODO: modify self.self.conv2d_1
        # intra-class DA ga-h2 & skip-conn-hd3 -> hd2
        self.intra_da_hd2 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[1], self.UpChannels, resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[1] + self.UpChannels,
            self.UpChannels
        )
        # TODO: modify self.conv1d_1
        # disagreement attention between h1 and hd1
        # maybe this attention is not necessary... test it
        self.intra_da_hd1 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[0], self.UpChannels, resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[0] + self.UpChannels,
            self.UpChannels
        )
        self.initialize_weights(
            kwargs.get('init_type'), layers_cls=(torch.nn.Conv2d, kwargs.get('batchnorm_cls')))

    def forward_1(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        h1 = self.conv1(x)  # h1->320*320*64

        return {'h1': h1}

    def forward_2(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h2'] = self.maxpool1(x['h1'])
        x['h2'] = self.conv2(x['h2'])  # h2->160*160*128

        return x

    def forward_3(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h3'] = self.maxpool2(x['h2'])
        x['h3'] = self.conv3(x['h3'])  # h3->80*80*256

        return x

    def forward_4(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h4'] = self.maxpool3(x['h3'])
        x['h4'] = self.conv4(x['h4'])  # h4->40*40*512

        return x

    def forward_5(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h5'] = self.maxpool4(x['h4'])
        x['hd5'] = self.conv5(x['h5'])  # h5->20*20*1024

        return x

    def forward_6(self, x: dict):
        assert isinstance(x, dict), type(x)

        return x

    def forward_7(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_PT_hd4'] = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(x['h1']))))
        x['h1_PT_hd4'] = apply_padding(x['h1_PT_hd4'], x['h4'])
        x['h2_PT_hd4'] = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(x['h2']))))
        x['h2_PT_hd4'] = apply_padding(x['h2_PT_hd4'], x['h4'])
        x['h3_PT_hd4'] = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(x['h3']))))
        x['h3_PT_hd4'] = apply_padding(x['h3_PT_hd4'], x['h4'])
        x['h4_Cat_hd4'] = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(x['h4'])))
        x['hd5_UT_hd4'] = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(
            self.hd5_UT_hd4(x['hd5']))))
        x['hd5_UT_hd4'] = apply_padding(x['hd5_UT_hd4'], x['h4'])
        x['hd4'] = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((x['h1_PT_hd4'], x['h2_PT_hd4'], x['h3_PT_hd4'], x['h4_Cat_hd4'], x['hd5_UT_hd4']), 1)
        )))  # hd4->40*40*UpChannels
        x['hd4'] = self.intra_da_hd4(x['hd4'], x['h4'], x['hd5'])

        return x

    def forward_8(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_PT_hd3'] = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(x['h1']))))
        x['h1_PT_hd3'] = apply_padding(x['h1_PT_hd3'], x['h3'])
        x['h2_PT_hd3'] = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(x['h2']))))
        x['h2_PT_hd3'] = apply_padding(x['h2_PT_hd3'], x['h3'])
        x['h3_Cat_hd3'] = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(x['h3'])))
        x['hd4_UT_hd3'] = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(
            self.hd4_UT_hd3(x['hd4']))))
        x['hd4_UT_hd3'] = apply_padding(x['hd4_UT_hd3'], x['h3'])
        x['hd5_UT_hd3'] = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(
            self.hd5_UT_hd3(x['hd5']))))
        x['hd5_UT_hd3'] = apply_padding(x['hd5_UT_hd3'], x['h3'])
        x['hd3'] = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((x['h1_PT_hd3'], x['h2_PT_hd3'], x['h3_Cat_hd3'], x['hd4_UT_hd3'], x['hd5_UT_hd3']), 1)
        )))  # hd3->80*80*UpChannels
        x['hd3'] = self.intra_da_hd3(x['hd3'], x['h3'], x['hd4'])

        return x

    def forward_9(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_PT_hd2'] = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(x['h1']))))
        x['h1_PT_hd2'] = apply_padding(x['h1_PT_hd2'], x['h2'])
        x['h2_Cat_hd2'] = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(x['h2'])))
        x['hd3_UT_hd2'] = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(
            self.hd3_UT_hd2(x['hd3']))))
        x['hd3_UT_hd2'] = apply_padding(x['hd3_UT_hd2'], x['h2'])
        x['hd4_UT_hd2'] = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(
            self.hd4_UT_hd2(x['hd4']))))
        x['hd4_UT_hd2'] = apply_padding(x['hd4_UT_hd2'], x['h2'])
        x['hd5_UT_hd2'] = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(
            self.hd5_UT_hd2(x['hd5']))))
        x['hd5_UT_hd2'] = apply_padding(x['hd5_UT_hd2'], x['h2'])
        x['hd2'] = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((x['h1_PT_hd2'], x['h2_Cat_hd2'], x['hd3_UT_hd2'], x['hd4_UT_hd2'], x['hd5_UT_hd2']), 1)
        )))  # hd2->160*160*UpChannels
        x['hd2'] = self.intra_da_hd2(x['hd2'], x['h2'], x['hd3'])

        return x

    def forward_10(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_Cat_hd1'] = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(x['h1'])))
        x['hd2_UT_hd1'] = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(
            self.hd2_UT_hd1(x['hd2']))))
        x['hd2_UT_hd1'] = apply_padding(x['hd2_UT_hd1'], x['h1'])
        x['hd3_UT_hd1'] = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(
            self.hd3_UT_hd1(x['hd3']))))
        x['hd3_UT_hd1'] = apply_padding(x['hd3_UT_hd1'], x['h1'])
        x['hd4_UT_hd1'] = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(
            self.hd4_UT_hd1(x['hd4']))))
        x['hd4_UT_hd1'] = apply_padding(x['hd4_UT_hd1'], x['h1'])
        x['hd5_UT_hd1'] = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(
            self.hd5_UT_hd1(x['hd5']))))
        x['hd5_UT_hd1'] = apply_padding(x['hd5_UT_hd1'], x['h1'])
        x['hd1'] = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((x['h1_Cat_hd1'], x['hd2_UT_hd1'], x['hd3_UT_hd1'], x['hd4_UT_hd1'], x['hd5_UT_hd1']), 1)
        )))  # hd1->320*320*UpChannels
        x['hd1'] = self.intra_da_hd1(x['hd1'], x['h1'], x['hd2'])

        return x

    def forward_11(self, x: dict):
        assert isinstance(x, dict), type(x)

        d1 = self.outconv1(x['hd1'])  # d1->320*320*n_classes

        return d1

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        x = self.forward_1(x)
        x = self.forward_2(x)
        x = self.forward_3(x)
        x = self.forward_4(x)
        x = self.forward_5(x)
        x = self.forward_6(x)
        x = self.forward_7(x)
        x = self.forward_8(x)
        x = self.forward_9(x)
        x = self.forward_10(x)
        x = self.forward_11(x)

        return x
