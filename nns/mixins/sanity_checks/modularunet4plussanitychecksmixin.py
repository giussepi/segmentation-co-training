# -*- coding: utf-8 -*-
""" nns/mixins/sanity_checks/modularunet4plussanitychecksmixin.py """

from typing import Optional

import torch
import torcheck
from torch.optim.optimizer import Optimizer

from nns.mixins.sanity_checks.base import BaseSanityChecksMixin


__all__ = ['ModularUNet4PlusSanityChecksMixin']


class ModularUNet4PlusSanityChecksMixin(BaseSanityChecksMixin):
    """
    Provides standard methods to add and manage sanity checks

    Usage:
        class SomeClass(ModularUNet4PlusSanityChecksMixin):
            def training(self):
                optimizer = ...
                if self.sanity_checks:
                    self.add_sanity_checks(optimizer)

                for epoch in range(num_epochs):
                    ...

            def validation(self):
                ...
                self.model.eval()
                if self.sanity_checks:
                    self.disable_sanity_checks()
                ...
                self.model.train()
                if self.sanity_checks:
                    self.enable_sanity_checks()
                ...
    """

    def add_sanity_checks(self, optimizer: Optimizer, model: Optional[torch.nn.Module] = None):
        """
        Adds model sanity checks.

        Note: overwrite this method as necessary
        See https://github.com/pengyan510/torcheck

        Kwargs:
            optimizer   <Optimizer>: Optimizer instance
            model <torch.nn.Module>: Module instance. Default self.model
        """
        assert isinstance(optimizer, Optimizer), type(optimizer)
        model = model if model is not None else self.model
        assert isinstance(model, torch.nn.Module), type(model)

        torcheck.register(optimizer)

        torcheck.add_module(
            model,
            module_name="my_model",
            changing=True,
            # output_range=(0, 1),
            # negate_range=False,
            check_nan=False,
            check_inf=False,
        )
