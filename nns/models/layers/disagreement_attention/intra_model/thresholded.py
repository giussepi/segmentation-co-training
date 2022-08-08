# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/intra_model/thresholded.py """

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod

from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock


__all__ = ['ThresholdedDABlock']


class ThresholdedDABlock(BaseDisagreementAttentionBlock):
    r"""
    Calculates the Thresholded Disagreement Attention and returns the act1 with the computed attention
    """

    def __init__(
            self, m1_act: int, m2_act: int,  /, *, n_channels: int = -1,
            batchnorm_cls: Optional[_BatchNorm] = None,
            init_type=UNet3InitMethod.KAIMING, upsample: bool = True,
            data_dimensions: int = 2,
            thresholds: Optional[Tuple[float]] = None, beta=-1.
    ):
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
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            init_type      <int>: Initialization method id.
                                  See gtorch_utils.nns.models.segmentation.unet3_plus.constants.UNet3InitMethod
                                  Default UNet3InitMethod.KAIMING
            upsample <bool>: Whether or not upsample the computed attention before applying the hadamard
                             product. Default True
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
            thresholds   <tuple>: Tuple with the lower and upper disagreement thresholds. If not value is
                                  provided is is set to (.25, .8). Default = None
            beta     <float>: user-defined boost for the activations boost rule. beta in ]-inf, 1[.
                                    Default -1. Note: when beta = 0 , it will work like the
                                    PureDABlock using the forward option 1
        """
        super().__init__(
            m1_act, m2_act, n_channels=n_channels, batchnorm_cls=batchnorm_cls, init_type=init_type,
            data_dimensions=data_dimensions
        )
        assert isinstance(upsample, bool), type(upsample)
        assert float('-inf') < beta < 1, beta
        if thresholds is not None:
            assert isinstance(thresholds, tuple), type(thresholds)
            assert len(thresholds) == 2, 'thresholds must contain only 2 values'
            assert 0 < thresholds[0] < 1, f'{thresholds[0]} is not in range ]0,1['
            assert 0 < thresholds[1] < 1, f'{thresholds[1]} is not in range ]0,1['
            assert thresholds[0] < thresholds[1], f'{thresholds[0]} must be less than {thresholds[1]}'

        self.thresholds = (.25, .8) if thresholds is None else thresholds
        self.upsample = upsample
        self.beta = beta

        convxd = nn.Conv2d if self.data_dimensions == 2 else nn.Conv3d
        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        if self.upsample:
            self.w1 = nn.Sequential(
                # modified following original implementation
                # convxd(m1_act, self.n_channels, kernel_size=1, stride=2, padding=0, bias=True),
                convxd(m1_act, self.n_channels, kernel_size=2, stride=2, padding=0, bias=False),
                # self.batchnorm_cls(self.n_channels)  # not present in original implementation
            )
            self.up = torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        else:
            self.w1 = nn.Sequential(
                # modified following original implementation
                # convxd(m1_act, self.n_channels, kernel_size=1, stride=2, padding=0, bias=True),
                convxd(m1_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False),
                # self.batchnorm_cls(self.n_channels)  # not present in original implementation
            )
        self.w2 = nn.Sequential(
            convxd(m2_act, self.n_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # self.batchnorm_cls(self.n_channels)  # not present in original implementation
        )
        self.act_with_attention = nn.Sequential(
            nn.ReLU(),
            convxd(self.n_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # self.batchnorm_cls(1),  # not present in original implementation
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            convxd(m1_act, m1_act, kernel_size=1, stride=1, padding=0, bias=True),
            self.batchnorm_cls(m1_act)
        )

    def activation_boost_rule(
            self, disagreement_matrix: torch.Tensor, feature_disagreement_idx: torch.Tensor):
        """
        Kwargs:
            disagreement_matrix <torch.Tensor>: difference between act2 and act1
            feature_disagreement_idx <torch.Tensor>: boolean tensor used to select the values to be
                                                boosted. It must be a boolean tensor so it can be used
                                                like this disagreement_matrix[feature_disagreement_idx]

        Returns:
            boosted_activations_maps <torch.Tensor>
        """
        assert isinstance(disagreement_matrix, torch.Tensor), type(disagreement_matrix)
        assert isinstance(feature_disagreement_idx, torch.Tensor), \
            type(feature_disagreement_idx)
        assert feature_disagreement_idx.dtype == torch.bool, feature_disagreement_idx.dtype

        if 0 <= self.beta < 1:
            disagreement_matrix[feature_disagreement_idx] *= (1 + self.beta)
        else:
            disagreement_matrix[~feature_disagreement_idx] = 0

        return disagreement_matrix

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
        feature_disagreement_idx = (torch.sigmoid(wact2) > self.thresholds[1]) * \
            (torch.sigmoid(wact1) < self.thresholds[0])
        # option 1 ############################################################
        delta = wact2 - wact1
        # option 2 ############################################################
        # delta = wact2 + torch.relu(wact2 - wact1)

        boosted_delta = self.activation_boost_rule(delta, feature_disagreement_idx)
        attention = self.act_with_attention(boosted_delta)

        if self.upsample:
            attention = self.up(attention)

        skip_with_attention = act1 * attention
        skip_with_attention = self.output(skip_with_attention)

        return skip_with_attention, attention
