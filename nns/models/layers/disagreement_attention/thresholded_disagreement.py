# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/thresholded_disagreement """

from typing import Tuple

import torch
from torch import nn

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['ThresholdedDisagreementAttentionBlock']


class ThresholdedDisagreementAttentionBlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the thresholded-disagreement attention from activations2 (belonging to model 2)
    towards activations1 (belonging to model 1) and returns activations1 with the computed attention

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
            m1_act         <int>: number of feature maps (channels) from model 1 (current model)
            m2_act         <int>: number of feature maps (channels) from model 2 (other model)
            n_channels     <int>: number of channels used during the calculations
                                  If not provided will be set to m1_act. Default -1
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

    def forward(self, skip_connection: torch.Tensor, act: torch.Tensor):
        """
        Calculates the attention using the difference between act and skip_connection. Finally,
        returns the skip_connection with the computed attention; i.e. the skip connection with
        the regions where act is better than skip highlighted

        Kwargs:
            skip_connection <torch.Tensor>: activations maps from the other model (it will receive
                                            the attention)
            act             <torch.Tensor>: activations maps from current model

        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wskip = self.w1(skip_connection)
        wact = self.w2(act)
        # FIXME: this line for now works but need to be updated
        resampled_wact = self.resample(wact)
        delta_phi2 = resampled_wact - wskip  # values where act is better than skip
        # delta_phi2 = resampled_wact * torch.relu(delta_phi2)  # opt3
        delta_phi2 = resampled_wact + torch.relu(delta_phi2)  # opt2
        # delta_phi2 = resampled_wact * (torch.relu(delta_phi2)+1)  # opt1
        psi2 = (torch.sigmoid(resampled_wact) > self.thresholds[1]) * \
            (torch.sigmoid(wskip) < self.thresholds[0])

        if self.beta >= 0:
            delta_phi2[psi2] *= (1+self.beta)
        else:
            delta_phi2[~psi2] = 0

        attention = self.attention_2to1(delta_phi2)
        skip_with_attention = skip_connection * attention

        return skip_with_attention, attention

    def forward_3(self, skip_connection: torch.Tensor, act: torch.Tensor):
        """
        Calculates the attention using the difference between skip_connection and act. Finally,
        returns the skip_connection with the computed attention; i.e. the skip connection with
        the regions where skip is better than act highlighted

        Kwargs:
            skip_connection <torch.Tensor>: activations maps from the other model (it will receive
                                            the attention)
            act             <torch.Tensor>: activations maps from current model

        Returns:
            skip_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        wskip = self.w1(skip_connection)
        wact = self.w2(act)
        # FIXME: this line for now works but need to be updated
        resampled_wact = self.resample(wact)
        delta_phi2 = wskip - resampled_wact  # values where wskip is better than wact
        # delta_phi2 = resampled_wact * torch.relu(delta_phi2)  # opt3
        delta_phi2 = wskip + torch.relu(delta_phi2)  # opt2
        # delta_phi2 = resampled_wact * (torch.relu(delta_phi2)+1)  # opt1
        psi2 = (torch.sigmoid(wskip) > self.thresholds[1]) * \
            (torch.sigmoid(resampled_wact) < self.thresholds[0])

        if self.beta >= 0:
            delta_phi2[psi2] *= (1+self.beta)
        else:
            delta_phi2[~psi2] = 0

        attention = self.attention_2to1(delta_phi2)
        skip_with_attention = skip_connection * attention

        return skip_with_attention, attention
