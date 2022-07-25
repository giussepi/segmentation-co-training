# -*- coding: utf-8 -*-
""" nns/models/mixins/init_mixin """

from typing import Any, Tuple

import torch

from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.nns.models.segmentation.unet3_plus.init_weights import init_weights


class InitMixin:
    """
    Mixing holding basic methods used during the model initialization
    NOTE: It must be added to your main neuronal network class

    Usage:
        MyModel(nn.Module, InitMixin):
            def __init__(self):
                ...
                self.initialize_weights()
                ...
    """

    def initialize_weights(self, init_type: int = UNet3InitMethod.KAIMING, /, *,
                           layers_cls: Tuple[Any] = None):
        """
        Validates the init_type and initialises the layers that are instances of any class from layers_cls

        Kwargs:
            init_type <int>: Initialization type.
                             see gtorch_utils.nns.models.segmentation.unet3_plus.constants.UNet3InitMethod
                             Default UNet3InitMethod.KAIMING
            layers_cls <tuple>: Tuple containing the layers classes to be initialized.
                             When not provided, it is set to (torch.nn.Conv1d, torch.nn.BatchNorm1d,
                             torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Conv3d, torch.nn.BatchNorm3d)
                             Default None
        """
        UNet3InitMethod.validate(init_type)

        if layers_cls is not None:
            assert isinstance(layers_cls, tuple), type(layers_cls)
            assert len(layers_cls) > 0, 'layers_cls cannot be empty'
        else:
            layers_cls = (
                torch.nn.Conv1d, torch.nn.BatchNorm1d,
                torch.nn.Conv2d, torch.nn.BatchNorm2d,
                torch.nn.Conv3d, torch.nn.BatchNorm3d
            )

        for module in self.modules():
            if isinstance(module, layers_cls):
                init_weights(module, init_type=init_type)
