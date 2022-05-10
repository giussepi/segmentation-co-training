# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/base_disagreement """

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
            self, m1_act: int, m2_act: int, /, *, n_channels: int = -1, resample: object = None, **kwargs):
        """
        Validates the arguments and initializes the attributes self.n_channels and self.resample properly

        Kwargs:
            m1_act         <int>: number of feature maps (channels) from model 1
            m2_act         <int>: number of feature maps (channels) from model 2
            n_channels     <int>: number of channels used during the calculations
                                  If not provided will be set to m1_act. Default -1
            resample <nn.Module>: Resample operation to be applied to activations2 to match activations1
                                  (e.g. identity, pooling, strided convolution, upconv, etc).
                                  Default nn.Identity()
        """
        super().__init__()
        assert isinstance(m1_act, int), type(m1_act)
        assert isinstance(m2_act, int), type(m2_act)
        assert isinstance(n_channels, int), type(n_channels)
        assert isinstance(resample, object), 'resample must be an instance'
        resample = resample if resample else nn.Identity()
        assert isinstance(resample, object)
        assert issubclass(resample.__class__, nn.Module)

        self.n_channels = m1_act if n_channels == -1 else n_channels
        self.resample = resample

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        raise NotImplementedError
