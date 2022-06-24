# -*- coding: utf-8 -*-
""" nns/models/x_attention_unet """

from typing import Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from gtorch_utils.nns.models.segmentation.unet.unet_parts import DoubleConv, Down, OutConv
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod

from nns.models.layers.disagreement_attention import AttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.layers import AttentionConvBlock
from nns.models.mixins import InitMixin


__all__ = ['XAttentionUNet']


class XAttentionUNet(torch.nn.Module, InitMixin):
    """
    Attention UNet with a replaceable attention unit

    Inspired on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    """

    def __init__(
            self, n_channels: int, n_classes: int, bilinear: bool = True,
            batchnorm_cls: _BatchNorm = torch.nn.BatchNorm2d, init_type=UNet3InitMethod.KAIMING,
            da_block_cls: BaseDisagreementAttentionBlock = AttentionBlock,
            da_block_config: Optional[dict] = None
    ):
        """
        Initializes the object instance

        Kwargs:
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes
            bilinear        <bool>: Whether or not use the bilinear mode
            batchnorm_cls <_BatchNorm>: batch normalization class
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
            da_block_cls <BaseDisagreementAttentionBlock>: Attention block to be used.
                                    Default AttentionBlock (the standard attention for Unet)
            da_block_config <dict>: Configuration for the instance to be created from da_block_cls
        """
        super().__init__()
        # attributes initalization ############################################
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type
        self.da_block_cls = da_block_cls
        if da_block_config:
            assert isinstance(da_block_config, dict), type(da_block_config)
            self.da_block_config = da_block_config
        else:
            self.da_block_config = {}

        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.bilinear, bool), type(self.bilinear)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)

        assert issubclass(self.da_block_cls, BaseDisagreementAttentionBlock), \
            f'{self.da_block_cls} is not a descendant of BaseDisagreementAttentionBlock'
        # adding extra configuration to da_block_config
        self.da_block_config['batchnorm_cls'] = self.batchnorm_cls
        self.da_block_config['init_type'] = self.init_type

        self.filters = [64, 128, 256, 512, 1024]

        # Encoder layers ######################################################
        self.inc = DoubleConv(n_channels, self.filters[0], batchnorm_cls=self.batchnorm_cls)
        self.down1 = Down(self.filters[0], self.filters[1], self.batchnorm_cls)
        self.down2 = Down(self.filters[1], self.filters[2], self.batchnorm_cls)
        self.down3 = Down(self.filters[2], self.filters[3], self.batchnorm_cls)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(self.filters[3], self.filters[4] // factor, self.batchnorm_cls)

        # Decoder layers ######################################################
        # intra-class DA skip-con down3 & gating signal down4 -> up1
        self.up1_with_da = AttentionConvBlock(
            # attention to skip_connection
            self.da_block_cls(self.filters[3], self.filters[4] // 2,
                              # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                              **self.da_block_config),
            2*self.filters[3],
            self.filters[3] // factor,
            batchnorm_cls=self.batchnorm_cls,
            bilinear=self.bilinear
        )
        # intra-class DA skip conn. down2 & gating signal up1_with_da -> up2
        self.up2_with_da = AttentionConvBlock(
            # attention to skip_connection
            self.da_block_cls(self.filters[2], self.filters[3] // 2,
                              # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                              **self.da_block_config),
            2*self.filters[2],
            self.filters[2] // factor,
            batchnorm_cls=self.batchnorm_cls,
            bilinear=self.bilinear
        )
        # intra-class DA skip conn. down1 & gating signal up2_with_da -> up3
        self.up3_with_da = AttentionConvBlock(
            # attention to skip_connection
            da_block_cls(self.filters[1], self.filters[2] // 2,
                         # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                         **self.da_block_config),
            2*self.filters[1],
            self.filters[1] // factor,
            batchnorm_cls=self.batchnorm_cls,
            bilinear=self.bilinear
        )
        # intra-class DA skip conn. inc & gating signal up_3_with_da -> up4
        # TODO: not sure if this last DA attention block is necessary
        self.up4_with_da = AttentionConvBlock(
            # attention to skip_connection
            da_block_cls(self.filters[0], self.filters[1] // 2,
                         # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                         **self.da_block_config),
            2*self.filters[0],
            self.filters[0],
            batchnorm_cls=self.batchnorm_cls,
            bilinear=self.bilinear
        )
        self.outc = OutConv(self.filters[0], n_classes)

        # initializing weights ################################################
        self.initialize_weights(self.init_type, layers_cls=(torch.nn.Conv2d, self.batchnorm_cls))

    def forward(self, x):
        # encoder #############################################################
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder #############################################################
        x = self.up1_with_da(x5, x4)
        x = self.up2_with_da(x, x3)
        x = self.up3_with_da(x, x2)
        x = self.up4_with_da(x, x1)
        logits = self.outc(x)

        return logits
