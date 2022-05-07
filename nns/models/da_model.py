# -*- coding: utf-8 -*-
""" nns/models/da_model """

import torch


__all__ = ['BaseDATrain']


class BaseDATrain(torch.nn.Module):
    """
    Base disagreement attention model

    Usage:
        MyDAModelMGR(BaseDAModel):
            def forward(self, model1_cls, kwargs1, model2_cls, kwargs2):
                  ...
    """

    def __init__(
            self, *, model1_cls: torch.nn.Module, kwargs1: dict, model2_cls: torch.nn.Module, kwargs2: dict):
        """
        Kwargs:
            model1_cls <torch.nn.Module>: instance of a model with disagreement attention
            kwargs1               <dict>: keyword arguments for model1
            model2_cls <torch.nn.Module>: instance of a model with disagreement attention
            kwargs2               <dict>: keyword arguments for model2
        """
        super().__init__()
        assert issubclass(model1_cls, torch.nn.Module), 'class model1 with disagreement attention'
        assert issubclass(model2_cls, torch.nn.Module), 'class model2 with disagreement attention'
        if kwargs1 is not None:
            assert isinstance(kwargs1, dict), type(kwargs1)
        if kwargs2 is not None:
            assert isinstance(kwargs2, dict), type(kwargs2)

        self.model1 = model1_cls(**kwargs1)
        self.model2 = model2_cls(**kwargs2)

    def forward(self, x: torch.Tensor, /, **kwargs):
        raise NotImplementedError(
            'You have to write a suitable forward pass to train model1 and model together')
