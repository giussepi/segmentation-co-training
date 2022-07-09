# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/intra_class/standard """

import torch
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
# from gtorch_utils.nns.models.segmentation.unet3_plus.init_weights import init_weights
from torch import nn

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['AttentionBlock']


class AttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the standard UNet attention and returns the act1 with the computed attention

    Usage:
        g = AttentionBlock(320, 1024, resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        g(act, gs)
    """

    def __init__(
            self, m1_act: int, m2_act: int,  /, *, n_channels: int = -1, batchnorm_cls=nn.BatchNorm2d,
            init_type=UNet3InitMethod.KAIMING, upsample: bool = True):
        """
        Initializes the object instance

        Kwargs:
            m1_act     <int>: number of feature maps (channels) from the activation which will
                              receive the attention
            m2_act     <int>: number of feature maps (channels) from the activation which will
                              be used to create the attention
            n_channels <int>: number of channels used during the calculations
                              If not provided will be set to max(m1_act, m2_act).
                              Default -1
            # FIXME: depending on how well the new forward methods works this resample logic coulb need
                     to be changed
            batchnorm_cls <_BatchNorm>: Batch normalization class to be used.
                                  Default nn.BatchNorm2d
            init_type      <int>: Initialization method id.
                                  See gtorch_utils.nns.models.segmentation.unet3_plus.constants.UNet3InitMethod
                                  Default UNet3InitMethod.KAIMING
            upsample <bool>: Whether or not upsample the computed attention before applying the hadamard
                             product. Default True
        """
        super().__init__(
            m1_act, m2_act, n_channels=n_channels, batchnorm_cls=batchnorm_cls, init_type=init_type)
        assert isinstance(upsample, bool), type(upsample)
        self.upsample = upsample

        if self.upsample:
            self.w1 = nn.Sequential(
                # modified following original implementation
                # nn.Conv2d(m1_act, self.n_channels, kernel_size=1, stride=2, padding=0, bias=True),
                nn.Conv2d(m1_act, self.n_channels, kernel_size=2, stride=2, padding=0, bias=False),
                # self.batchnorm_cls(self.n_channels)  # not present in original implementation
            )
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.w1 = nn.Sequential(
                # modified following original implementation
                # nn.Conv2d(m1_act, self.n_channels, kernel_size=1, stride=2, padding=0, bias=True),
                nn.Conv2d(m1_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False),
                # self.batchnorm_cls(self.n_channels)  # not present in original implementation
            )
        self.w2 = nn.Sequential(
            nn.Conv2d(m2_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # self.batchnorm_cls(self.n_channels)  # not present in original implementation
        )
        self.act_with_attention = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # self.batchnorm_cls(1),  # not present in original implementation
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Conv2d(m1_act, m1_act, kernel_size=1, stride=1, padding=0, bias=True),
            self.batchnorm_cls(m1_act)
        )

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, self.batchnorm_cls)):
        #         init_weights(m, init_type=self.init_type)

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Kwargs:
            act1 <torch.Tensor>: activations maps which will receive the attention
            act2 <torch.Tensor>: activations maps employed to calculate the attention

        Returns:
            act1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        assert isinstance(act1, torch.Tensor), type(act1)
        assert isinstance(act2, torch.Tensor), type(act2)

        wact1 = self.w1(act1)
        wact2 = self.w2(act2)
        attention = self.act_with_attention(wact1+wact2)

        if self.upsample:
            attention = self.up(attention)

        skip_with_attention = act1 * attention
        skip_with_attention = self.output(skip_with_attention)

        return skip_with_attention, attention
