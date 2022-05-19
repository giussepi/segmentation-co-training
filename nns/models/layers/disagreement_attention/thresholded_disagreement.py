# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/thresholded_disagreement """

from typing import Tuple

import torch
from torch import nn

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['ThresholdedDisagreementAttentionBlock']


class ThresholdedDisagreementAttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the thresholded-disagreement attention between activations2
    and activations1 and returns activations1 with the computed attention

    \begin{equation}
    \begin{split}
    \Phi_2^R &= \mathcal{R}_{2\rightarrow 1}(\Phi_2) \\
    \Delta \Phi_{2} &= \Phi_2^R - \Phi_{1} \\
    \Psi_{_2} &= (\sigma_s(\Phi_2^R) > \tau_u) \oslash (\sigma_s(\Phi_{1}) < \tau_l) \\
    \Delta \Phi_{2}[\Psi_{_2}] *&= (1 + \beta) ~\text{or set $\Delta \Phi_{2}[\sim\Psi_{_2}]=0$}\\
    \underline{A_{2\rightarrow 1}} &= \underline{\sigma_s(\rm{\bowtie_{1, 1\times1}}(\Delta \Phi_{2}))} \\
    \Phi_1^{A} &= \Phi_{1} \oslash A_{2\rightarrow 1} \\
    \end{split}
    \end{equation}
    """

    def __init__(
            self, m1_act: int, m2_act: int, /, *, n_channels: int = -1, resample: object = None,
            thresholds: Tuple[float] = None, beta: float = -1.0
    ):
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
            resample    <object>: Resample operation to be applied to activations2 to match activations1
                                  (e.g. identity, pooling, strided convolution, upconv, etc).
                                  Default nn.Identity()
            thresholds   <tuple>: Tuple with the lower and upper disagreement thresholds. If not value is
                                  provided is is set to (.25, .8). Default = None
            beta         <floar>: User-defined attention boost in range ]0,1[. Set it to a negative value
                                  to not use it and set all values not included in the feature disagreement
                                  index psi2 to zero. Default -1.0
        """
        super().__init__(m1_act, m2_act, n_channels=n_channels, resample=resample)

        if thresholds is not None:
            assert isinstance(thresholds, tuple), type(thresholds)
            assert len(thresholds) == 2, 'thresholds must contain only 2 values'
            assert 0 < thresholds[0] < 1, f'{thresholds[0]} is not in range ]0,1['
            assert 0 < thresholds[1] < 1, f'{thresholds[1]} is not in range ]0,1['
            assert thresholds[0] < thresholds[1], f'{thresholds[0]} must be less than {thresholds[1]}'
        assert isinstance(beta, float), type(beta)
        if beta >= 0:
            assert 0 <= beta < 1, f'{beta} must be in range [0,1['

        self.thresholds = (.25, .8) if thresholds is None else thresholds
        self.beta = beta
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
        Calculates the attention using the difference between act2 and act1. Finally,
        returns the act1 with the computed attention; i.e. the act1 with
        the regions where act2 is better than act1 highlighted

        Kwargs:
            act1 <torch.Tensor>: activations maps which will receive the attention
            act2 <torch.Tensor>: activations maps employed to create the attention

        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wact1 = self.w1(act1)
        wact2 = self.w2(act2)
        # FIXME: this line for now works but need to be updated
        resampled_wact2 = self.resample(wact2)
        delta_phi2 = resampled_wact2 - wact1  # values where act2 is better than act1
        # delta_phi2 = resampled_wact2 * torch.relu(delta_phi2)  # opt3
        delta_phi2 = resampled_wact2 + torch.relu(delta_phi2)  # opt2
        # delta_phi2 = resampled_wact2 * (torch.relu(delta_phi2)+1)  # opt1
        psi2 = (torch.sigmoid(resampled_wact2) > self.thresholds[1]) * \
            (torch.sigmoid(wact1) < self.thresholds[0])

        if self.beta >= 0:
            delta_phi2[psi2] *= (1+self.beta)
        else:
            delta_phi2[~psi2] = 0

        attention = self.attention_2to1(delta_phi2)
        act1_with_attention = act1 * attention

        return act1_with_attention, attention

    def forward_3(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Calculates the attention using the difference between act1 and act2. Finally,
        returns the act1 with the computed attention; i.e. the act1 with
        the regions where act1 is better than act2 highlighted

        Kwargs:
            act1 <torch.Tensor>: activations maps which will receive the attention
            act2 <torch.Tensor>: activations maps employed to calculate the attention

        Returns:
            act1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wact1 = self.w1(act1)
        wact2 = self.w2(act2)
        # FIXME: this line for now works but need to be updated
        resampled_wact2 = self.resample(wact2)
        delta_phi2 = wact1 - resampled_wact2  # values where act1 is better than act2
        # delta_phi2 = resampled_wact2 * torch.relu(delta_phi2)  # opt3
        delta_phi2 = wact1 + torch.relu(delta_phi2)  # opt2
        # delta_phi2 = resampled_wact2 * (torch.relu(delta_phi2)+1)  # opt1
        psi2 = (torch.sigmoid(wact1) > self.thresholds[1]) * \
            (torch.sigmoid(resampled_wact2) < self.thresholds[0])

        if self.beta >= 0:
            delta_phi2[psi2] *= (1+self.beta)
        else:
            delta_phi2[~psi2] = 0

        attention = self.attention_2to1(delta_phi2)
        act1_with_attention = act1 * attention

        return act1_with_attention, attention
