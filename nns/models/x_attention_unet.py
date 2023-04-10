# -*- coding: utf-8 -*-
""" nns/models/x_attention_unet """

from typing import Optional

import torch
from logzero import logger
from gtorch_utils.nns.models.mixins import InitMixin
from gtorch_utils.nns.models.segmentation.unet.unet_parts import DoubleConv, Down, OutConv,\
    UpConcat, UnetDsv, UnetGridGatingSignal
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from torch.nn.modules.batchnorm import _BatchNorm

from nns.models.layers.disagreement_attention.intra_model import AttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.layers import AttentionConvBlock


__all__ = ['XAttentionUNet']


class XAttentionUNet(torch.nn.Module, InitMixin):
    """
    Attention UNet with a replaceable attention unit

    Based on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py and
    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_CT_single_att_dsv_3D.py
    """

    def __init__(
            self, n_channels: int, n_classes: int, bilinear: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, init_type=UNet3InitMethod.KAIMING,
            data_dimensions: int = 2,
            da_block_cls: BaseDisagreementAttentionBlock = AttentionBlock,
            da_block_config: Optional[dict] = None,
            dsv: bool = True
    ):
        """
        Initializes the object instance

        Kwargs:
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes
            bilinear        <bool>: Whether or not use the bilinear mode
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
            da_block_cls <BaseDisagreementAttentionBlock>: Attention block to be used.
                                    Default AttentionBlock (the standard attention for Unet)
            da_block_config <dict>: Configuration for the instance to be created from da_block_cls.
                                    DO NOT PROVIDE 'n_channels' or 'data_dimensions' IN THIS DICTIONARY.
                                    They will be removed because they are properly defined per
                                    AttentionConvBlock. Default None
             dsv            <bool>: Whether or not apply deep supervision. Default True
        """
        super().__init__()

        # attributes initalization ############################################
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.data_dimensions = data_dimensions
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type
        self.da_block_cls = da_block_cls
        self.dsv = dsv

        if da_block_config:
            assert isinstance(da_block_config, dict), type(da_block_config)
            self.da_block_config = da_block_config
        else:
            self.da_block_config = {}

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.bilinear, bool), type(self.bilinear)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)
        assert issubclass(self.da_block_cls, BaseDisagreementAttentionBlock), \
            f'{self.da_block_cls} is not a descendant of BaseDisagreementAttentionBlock'
        assert isinstance(self.dsv, bool), type(self.dsv)

        # adding extra configuration to da_block_config
        self.da_block_config['batchnorm_cls'] = self.batchnorm_cls
        self.da_block_config['init_type'] = self.init_type

        # removing n_channels from da_block_config, because this value is already
        # defined per AttentionConvBlock
        if 'n_channels' in self.da_block_config:
            logger.warning(
                f'n_channels: {self.da_block_config["n_channels"]} has been removed from '
                'da_block_config'
            )
            self.da_block_config.pop('n_channels')

        if 'data_dimensions' in self.da_block_config:
            logger.warning(
                f'data_dimensions: {self.da_block_config["data_dimensions"]} has been removed from '
                'da_block_config'
            )
            self.da_block_config.pop('data_dimensions')

        self.filters = [64, 128, 256, 512, 1024]

        # Encoder layers ######################################################
        self.inc = DoubleConv(
            n_channels, self.filters[0], batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions
        )
        self.down1 = Down(self.filters[0], self.filters[1], self.batchnorm_cls, self.data_dimensions)
        self.down2 = Down(self.filters[1], self.filters[2], self.batchnorm_cls, self.data_dimensions)
        self.down3 = Down(self.filters[2], self.filters[3], self.batchnorm_cls, self.data_dimensions)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(
            self.filters[3], self.filters[4] // factor, self.batchnorm_cls, self.data_dimensions)  # centre
        self.gating = UnetGridGatingSignal(
            self.filters[4] // factor, self.filters[4] // factor, kernel_size=1,
            batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions
        )

        # Decoder layers ######################################################
        # intra-class DA skip-con down3 & gating signal down4 -> up1
        self.up1_with_da = AttentionConvBlock(
            # attention to skip_connection
            self.da_block_cls(self.filters[3], self.filters[4] // factor,
                              n_channels=self.filters[3],
                              data_dimensions=self.data_dimensions,
                              **self.da_block_config),
            self.filters[3] + (self.filters[4] // factor),
            self.filters[3] // factor,
            batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions,
        )
        # intra-class DA skip conn. down2 & gating signal up1_with_da -> up2
        self.up2_with_da = AttentionConvBlock(
            # attention to skip_connection
            self.da_block_cls(self.filters[2], self.filters[3] // factor,
                              n_channels=self.filters[2],
                              data_dimensions=self.data_dimensions,
                              **self.da_block_config),
            self.filters[2] + (self.filters[3] // factor),
            self.filters[2] // factor,
            batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions,
        )
        # intra-class DA skip conn. down1 & gating signal up2_with_da -> up3
        self.up3_with_da = AttentionConvBlock(
            # attention to skip_connection
            self.da_block_cls(self.filters[1], self.filters[2] // factor,
                              n_channels=self.filters[1],
                              data_dimensions=self.data_dimensions,
                              **self.da_block_config),
            self.filters[1] + (self.filters[2] // factor),
            self.filters[1] // factor,
            batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions,
        )
        self.up4 = UpConcat(
            self.filters[1] // factor, self.filters[0], self.bilinear, self.batchnorm_cls,
            self.data_dimensions
        )

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        if self.dsv:
            # deep supervision ################################################
            self.dsv1 = UnetDsv(
                in_size=self.filters[3] // factor, out_size=self.n_classes, scale_factor=8,
                data_dimensions=self.data_dimensions
            )
            self.dsv2 = UnetDsv(
                in_size=self.filters[2] // factor, out_size=self.n_classes, scale_factor=4,
                data_dimensions=self.data_dimensions
            )
            self.dsv3 = UnetDsv(
                in_size=self.filters[1] // factor, out_size=self.n_classes, scale_factor=2,
                data_dimensions=self.data_dimensions
            )
            self.dsv4 = convxd(in_channels=self.filters[0], out_channels=self.n_classes, kernel_size=1)
            self.outc = OutConv(self.n_classes*4, self.n_classes, self.data_dimensions)
        else:
            self.outc = OutConv(self.filters[0], self.n_classes, self.data_dimensions)

            # initializing weights ################################################
        self.initialize_weights(self.init_type, layers_cls=(convxd, self.batchnorm_cls))

    def forward(self, x):
        # encoder #############################################################
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        gating = self.gating(x5)

        # decoder #############################################################

        d1 = self.up1_with_da(x5, x4, central_gating=gating)
        d2 = self.up2_with_da(d1, x3)
        d3 = self.up3_with_da(d2, x2)
        d4 = self.up4(d3, x1)

        if self.dsv:
            # deep supervision ####################################################
            dsv1 = self.dsv1(d1)
            dsv2 = self.dsv2(d2)
            dsv3 = self.dsv3(d3)
            dsv4 = self.dsv4(d4)
            logits = self.outc(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        else:
            logits = self.outc(d4)

        return logits
