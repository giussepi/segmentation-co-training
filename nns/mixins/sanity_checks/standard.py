# -*- coding: utf-8 -*-
""" nns/mixins/sanity_checks/standard.py """

import torcheck
from torch.optim.optimizer import Optimizer

__all__ = ['SanityChecksMixin']


class SanityChecksMixin:
    """
    Provides standard methods to add and manage sanity checks

    Usage:
        class SomeClass(SanityChecksMixin):
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

    def add_sanity_checks(self, optimizer: Optimizer):
        """
        Adds model sanity checks.

        Note: overwrite this method as necessary
        See https://github.com/pengyan510/torcheck
        """
        assert isinstance(optimizer, Optimizer), type(optimizer)
        torcheck.register(optimizer)

        torcheck.add_module(
            self.model,
            module_name="my_model",
            changing=True,
            # output_range=(0, 1),
            # negate_range=False,
            check_nan=True,
            check_inf=True,
        )

    @staticmethod
    def disable_sanity_checks():
        torcheck.disable()

    @staticmethod
    def enable_sanity_checks():
        torcheck.enable()
