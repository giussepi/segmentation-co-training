# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/base_disagreement """

from typing import Callable, Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
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
            self, m1_act: int, m2_act: int, /, *, n_channels: int = -1, resample: Callable = None,
            batchnorm_cls: Optional[_BatchNorm] = None, init_type=UNet3InitMethod.KAIMING,
            data_dimensions: int = 2, **kwargs
    ):
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
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            init_type      <int>: Initialization method id.
                                  See gtorch_utils.nns.models.segmentation.unet3_plus.constants.UNet3InitMethod
                                  Default UNet3InitMethod.KAIMING
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.m1_act = m1_act
        self.m2_act = m2_act
        self.n_channels = max(self.m1_act, self.m2_act) if n_channels == -1 else n_channels
        self.resample = resample if resample is not None else nn.Identity()
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.m1_act, int), type(self.m1_act)
        assert isinstance(self.m2_act, int), type(self.m2_act)
        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert callable(self.resample), 'resample must be a callable'
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

    def forward(self, act1: torch.Tensor, act2: torch.Tensor):
        """
        Returns:
            activations1_with_attention <torch.Tensor>, attention <torch.Tensor>
        """
        raise NotImplementedError
