# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/merged_disagreement """

import torch
from torch import nn


__all__ = ['MergedDisagreementAttentionBlock']


class MergedDisagreementAttentionBlock(nn.Module):
    r"""
    Calculates the merged-disagreement attention from activations2 (belonging to model 2) towards
    activations1 (belonging to model 1) and returns activations1 with the computed attention

    \begin{equation}
    \begin{split}
    \Delta \Phi_1 &=  \left| \Phi_1 - \mathcal{R}_{2 \rightarrow 1}(\Phi_2) \right| \\
    \underline{A_{2\rightarrow 1}} &=  \underline{\sigma_s\left(\rm{\bowtie_{1,1\times1}}( \Delta \Phi_1 \right) ) } \\
    \Phi_1^A &= \Phi_1 \oslash A_{2\rightarrow 1} \\
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
                                  (e.g. identity, pooling, strided convolution, upconv, etc).
                                  Default nn.Identity()
        """
        super().__init__()
        assert isinstance(m1_act, int), type(m1_act)
        assert isinstance(m2_act, int), type(m2_act)
        assert isinstance(n_channels, int), type(n_channels)
        resample = resample if resample else nn.Identity()
        assert isinstance(resample, object), 'resample must be an instance'

        if n_channels == -1:
            n_channels = m1_act

        self.w1 = nn.Sequential(
            nn.Conv2d(m1_act, n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_channels)
        )

        self.w2 = nn.Sequential(
            nn.Conv2d(m2_act, n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_channels)
        )

        self.attention_2to1 = nn.Sequential(
            nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.resample = resample

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Kwargs:
            act1 <torch.Tensor>: activations maps from model 1 (gating signal)
            act2 <torch.Tensor>: activations maps from model 2 (skip connection)

        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wact1 = self.w1(act1)
        wact2 = self.w2(act2)
        delta_phi1 = torch.abs(wact1 - self.resample(wact2))
        attention = self.attention_2to1(delta_phi1)
        act1_with_attention = act1 * attention

        return act1_with_attention, attention
