# -*- coding: utf-8 -*-
""" nns/utils/sync_batchnorm/utils """

from torch import nn
from .batchnorm import SynchronizedBatchNorm2d


def get_batchnorm2d_class(num_gpus):
    """
    Returns the right BatchNorm2d based on the number of GPUS

    Args:
        num_gpus <int>: number of GPUS

    Returns:
        nn.BatchNorm2d or SynchronizedBatchNorm2d
    """
    assert isinstance(num_gpus, int), type(num_gpus)

    return nn.BatchNorm2d if num_gpus < 2 else SynchronizedBatchNorm2d
