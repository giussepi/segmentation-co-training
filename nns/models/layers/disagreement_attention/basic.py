# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/basic """

import torch
from torch import nn

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['AttentionBlock']


class AttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the attention and returns the skip connections with the computed attention

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
            self, skip_act: int, gs_act: int,  /, *, n_channels: int = -1, resample: object = None):
        """
        Initializes the object instance

        Kwargs:
            skip_act       <int>: number of feature maps (channels) of the skip connection
            gs_act         <int>: number of feature maps (channels) of the gating signal
            n_channels     <int>: number of channels for act  used during the calculations
                                  If not provided will be set to skip_act. Default -1
            resample    <object>: Resample operation to be applied to gs_act to match skip_act
                                  (e.g. identity, pooling, strided convolution, upconv, etc)
                                  Default nn.Identity()
        """
        if n_channels == -1:
            n_channels = skip_act

        super().__init__(skip_act, gs_act, n_channels=n_channels, resample=resample)

        self.w1 = nn.Sequential(
            nn.Conv2d(skip_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.n_channels)
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(gs_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.n_channels)
        )
        self.act_with_attention = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, skip_connection: torch.Tensor, gating_signal: torch.Tensor):
        """
        Kwargs:
            skip_connection  <torch.Tensor>: skip connection feature activations maps
                                             (where the attention will be applied)
            gating_signal    <torch.Tensor>: gating signal activations maps

        Returns:
            skip_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(gating_signal, torch.Tensor), type(gating_signal)

        wskip = self.w1(skip_connection)
        wgs = self.w2(gating_signal)
        wgs = self.resample(wgs)
        attention = self.act_with_attention(wskip+wgs)
        skip_with_attention = skip_connection * attention

        return skip_with_attention, attention
