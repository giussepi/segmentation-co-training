# -*- coding: utf-8 -*-
""" nns/utils/normalize """

import torch


__all__ = ['Normalizer']


class Normalizer:
    """
    Normalizes a tensor [batch, channels, height, width] by batch

    Usage:
        Normalizer()(mytensor)
    """

    def __call__(self, tensor: torch.Tensor, /, *, in_place: bool = False):
        return self.process(tensor, in_place=in_place)

    @staticmethod
    def process(tensor: torch.Tensor, /, *, in_place: bool = False):
        """
        Kwargs:
            tensor <torch.Tensor>: torch tensor with shape [batch, ...] to be normalized by batch
            in_place       <bool>: Whether or not perform the normalization in place.
                                   Requires a tensor of dtype torch.float or torch.double
                                   Default False
        """
        assert isinstance(tensor, torch.Tensor), type(tensor)
        assert isinstance(in_place, bool), type(in_place)

        if in_place:
            assert tensor.dtype in (torch.float, torch.double), (
                f'the provided tensor (dtype={tensor.dtype}) must be of type torch.float or '
                'torch.double to perform in_place normalization'
            )

        shape = tensor.size()

        if not in_place:
            tensor = tensor.clone()

            if tensor.dtype not in (torch.float, torch.double):
                tensor = tensor.float()

        tensor = tensor.view(shape[0], -1)
        tensor -= tensor.min(1, keepdim=True)[0]
        tensor /= tensor.max(1, keepdim=True)[0]
        tensor = tensor.view(shape)

        return tensor
