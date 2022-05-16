# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/pure_disagrement """

import torch
from torch import nn

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['PureDisagreementAttentionBlock']


class PureDisagreementAttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the pure-disagreement attention from activations2 (belonging to model 2) towards
    activations1 (belonging to model 1) and returns activations1 with the computed attention

    \begin{equation}
    \begin{split}
    \Delta \Phi_{2} &= \mathcal{R}_{2\rightarrow 1}(\Phi_{2}) - \Phi_{1} \\
    \underline{A_{2\rightarrow 1}} &= \underline{\sigma_s(\rm{\bowtie_{1, 1\times1}}(\sigma_r(\Delta \Phi_{2})))} \\
    \Phi_1^A &= \Phi_1 \oslash A_{2\rightarrow 1}
    \end{split}
    \end{equation}

    """

    def __init__(
            self, m1_act: int, m2_act: int, /, *, n_channels: int = -1, resample: object = None):
        """
        Initializes the object instance

        Kwargs:
            m1_act         <int>: number of feature maps (channels) from model 1
            m2_act         <int>: number of feature maps (channels) from model 2
            n_channels     <int>: number of channels used during the calculations
                                  If not provided will be set to m1_act. Default -1
            resample    <object>: Resample operation to be applied to activations2 to match activations1
                                  (e.g. identity, pooling, strided convolution, upconv, etc)
                                  Default nn.Identity()
        """
        super().__init__(m1_act, m2_act, n_channels=n_channels, resample=resample)

        self.w1 = nn.Sequential(
            nn.Conv2d(m1_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.n_channels)
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(m2_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.n_channels)
        )
        self.attention_2to1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Kwargs:
            act1 <torch.Tensor>: activations maps from model 1
            act2 <torch.Tensor>: activations maps from model 2 (skip connection)

        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wact1 = self.w1(act1)
        wact2 = self.w2(act2)
        delta_phi2 = self.resample(wact2) - wact1
        attention = self.attention_2to1(delta_phi2)
        act1_with_attention = act1 * attention

        return act1_with_attention, attention
