# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/merged_disagreement """

import torch
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from torch import nn
from typing import Callable

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['EmbeddedDisagreementAttentionBlock']


class EmbeddedDisagreementAttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the embedded disaggrement attention from activations2 (belonging to model 2) towards
    activations1 (belonging to model 1) and returns activations1 with the computed attention

    \begin{equation}
    \begin{split}
    \Phi^R_2 &= \mathcal{R}_{2 \rightarrow 1}(\Phi_2) \\
    \Phi_1^A &= \Phi^R_2 + \left| \Phi^R_2 - \Phi_1  \right| \text{ \# embedded disagreement of 2 into 1}\\
    \underline{A_{2 \rightarrow 1}} &= \underline{\sigma_s\left(\rm{\bowtie_{1,1\times1}}( \sigma_r(\Phi_1^A / \Phi_1 \right)) ) } \\
    \Phi^R_1 &= \mathcal{R}_{1 \rightarrow 2}(\Phi_1) \\
    \Phi_2^A &= \Phi^R_1 + \left| \Phi^R_1 - \Phi_2\right| \text{ \# embedded disagreement of 1 into 2}\\
    \underline{A_{1 \rightarrow 2}} &= \underline{\sigma_s\left(\rm{\bowtie_{1,1\times1}}( \sigma_r(\Phi_2^A / \Phi_2 \right)) ) } \\
    \end{split}
    \end{equation}
    """

    def __init__(
            self, m1_act: int, m2_act: int, /, *, n_channels: int = -1, resample: Callable = None,
            batchnorm_cls=nn.BatchNorm2d, init_type=UNet3InitMethod.KAIMING):
        """
        Initializes the object instance

        Kwargs:
            m1_act         <int>: number of feature maps (channels) from the activation which will
                                  receive the attention
            m2_act         <int>: number of feature maps (channels) from the activation which will
                                  be used to create the attention
            n_channels     <int>: number of channels used during the calculations
                                  If not provided will be set to max(m1_act, m2_act).
                                  Default -1
            # FIXME: depending on how well the new forward methods works this resample logic coulb need
                     to be changed
            resample  <Callable>: Resample operation to be applied to activations2 to match activations1
                                  (e.g. identity, pooling, strided convolution, upconv, etc).
                                  Default nn.Identity()
            batchnorm_cls <_BatchNorm>: Batch normalization class to be used.
                                  Default nn.BatchNorm2d
            init_type      <int>: Initialization method id.
                                  See gtorch_utils.nns.models.segmentation.unet3_plus.constants.UNet3InitMethod
                                  Default UNet3InitMethod.KAIMING
        """
        super().__init__(
            m1_act, m2_act, n_channels=n_channels, resample=resample, batchnorm_cls=batchnorm_cls,
            init_type=init_type
        )

        self.w1 = nn.Sequential(
            nn.Conv2d(m1_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            self.batchnorm_cls(self.n_channels)
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(m2_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            self.batchnorm_cls(self.n_channels)
        )
        self.attention_2to1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            self.batchnorm_cls(1),
            nn.Sigmoid()
        )

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Kwargs:
            act1 <torch.Tensor>: activations maps which will receive the attention
            act2 <torch.Tensor>: activations maps employed to calculate the attention

        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wact1 = self.w1(act1)
        wact2 = self.w2(act2)
        wact2 = self.resample(wact2)
        act1_with_attention = wact2 + torch.abs(wact2 - wact1)
        # FIXME: I cannot divide by act1 because at some point at some point
        # the number of channels from act1_with_attention and wact1 might be different
        # attention = self.attention_2to1(act1_with_attention/act1)
        # If want to do this
        attention = self.attention_2to1(act1_with_attention/wact1)
        # I will have to replace the DAConvBlock conv_in_channels
        # from self.filters[x]+self.UpChannels to 2*self.UpChannles
        # from self.intra_da_hd3 and onwards (in UNet_3Plus_Intra_DA)
        # OR FIND A BETTER SOLUTION (adding extra layers to process
        # the results could work...)

        return act1_with_attention, attention
