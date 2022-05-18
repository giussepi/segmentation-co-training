# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/base_disagreement """

from typing import Callable

import torch
from torch import nn


__all__ = ['BaseDisagreementAttentionBlock']


class BaseDisagreementAttentionBlock(nn.Module):
    """
    Base template with the main method's signatures for disagreement attention blocks

    Usage:
        class MyDAblock(BaseDisagreementAttentionBlock):
            def __init__(...):
                super().__init__(m1_act, m2_act, n_channels=n_channels, resample=resample)
                ...

            def forward(...):
                ...
    """

    def __init__(
            self, m1_act: int, m2_act: int, /, *, n_channels: int = -1, resample: Callable = None, **kwargs):
        """
        Validates the arguments and initializes the attributes self.n_channels and self.resample properly

        Kwargs:
            m1_act         <int>: number of feature maps (channels) from model 1
            m2_act         <int>: number of feature maps (channels) from model 2
            n_channels     <int>: number of channels used during the calculations
                                  If not provided will be set to max(m1_act, m2_act).
                                  Default -1
            resample  <Callable>: Resample operation to be applied to activations2 to match activations1
                                  (e.g. identity, pooling, strided convolution, upconv, etc).
                                  Default nn.Identity()
        """
        super().__init__()
        assert isinstance(m1_act, int), type(m1_act)
        assert isinstance(m2_act, int), type(m2_act)
        assert isinstance(n_channels, int), type(n_channels)
        resample = resample if resample else nn.Identity()
        assert callable(resample), 'resample must be a callable'

        self.n_channels = max(m1_act, m2_act) if n_channels == -1 else n_channels
        self.resample = resample

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        raise NotImplementedError
