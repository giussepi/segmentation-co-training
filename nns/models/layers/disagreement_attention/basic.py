# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/basic """

from typing import Callable

import torch
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
# from gtorch_utils.nns.models.segmentation.unet3_plus.init_weights import init_weights
from torch import nn

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['AttentionBlock']


class AttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the attention and returns the act1 with the computed attention

    \begin{equation}
    \begin{split}
    \Phi_{a} &= \rm{\bowtie_{g, 1\times1}}(\Phi_{a}) \\
    \Phi_{gs} &= \tex{Upsample}_{2}(\rm{\bowtie_{g, 1\times1}}(\Phi_{gs})) \\
    \underline{A} &= \underline{\sigma_s(\rm{\bowtie_{1, 1\times1}}(\sigma_r(\Phi_{a} + \Phi_{gs})))} \\
    \Phi_{a}^A &= \Phi_{a} \oslash A
    \end{split}
    \end{equation}

    \begin{align}
    \text{Where} ~\Phi_{a}: & ~\text{Feature maps to be updated with attention} \\
    \Phi_{gs}: & ~\text{Gating signal} \\
    g: & ~\text{Number of feature maps of the gating signal. (1024 at the beginning, then 320)} \\
    \bowtie_{x,1\times1}: & ~\text{2D convolution with $1\times1$ kernel and $x$ output channels} \\
    \sigma_s: & ~ \text{Sigmoid activation} \\
    \sigma_r: & ~ \text{ReLU activation} \\
    \oslash: & ~ \text{Hadamard product}\\
    \Phi_a^A: & ~\text{Feature maps with attention}
    \end{align}

    Usage:
        g = AttentionBlock(320, 1024, resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        g(act, gs)
    """

    def __init__(
            self, m1_act: int, m2_act: int,  /, *, n_channels: int = -1, resample: Callable = None,
            batchnorm_cls=nn.BatchNorm2d, init_type=UNet3InitMethod.KAIMING):
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
            resample <Callable>: Resample operation to be applied to activations2 to match activations1
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
        self.act_with_attention = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            self.batchnorm_cls(1),
            nn.Sigmoid()
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
        wact2 = self.resample(wact2)
        attention = self.act_with_attention(wact1+wact2)
        skip_with_attention = act1 * attention

        return skip_with_attention, attention
