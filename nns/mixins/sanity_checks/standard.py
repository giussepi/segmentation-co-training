# -*- coding: utf-8 -*-
""" nns/mixins/sanity_checks/standard.py """

from nns.mixins.sanity_checks.base import BaseSanityChecksMixin


__all__ = ['SanityChecksMixin']


class SanityChecksMixin(BaseSanityChecksMixin):
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
