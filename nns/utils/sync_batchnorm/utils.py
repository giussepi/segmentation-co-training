# -*- coding: utf-8 -*-
""" nns/utils/sync_batchnorm/utils.py """

import torch

from . import settings
from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d


def get_batchnormxd_class():
    """
    Returns the right BatchNorm based on the number of GPUS

    Returns:
        torch.nn.BatchNorm<1|2|3>d or SynchronizedBatchNorm<1|2|3>d
    """
    assert settings.DATA_DIMENSIONS in (1, 2, 3), 'Only 1D, 2D and 3D data is supported'

    if settings.CUDA == settings.MULTIGPUS == settings.PATCH_REPLICATION_CALLBACK == True and \
       torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if settings.DATA_DIMENSIONS == 1:
            return SynchronizedBatchNorm1d

        if settings.DATA_DIMENSIONS == 2:
            return SynchronizedBatchNorm2d

        return SynchronizedBatchNorm3d

    if settings.DATA_DIMENSIONS == 1:
        return torch.nn.BatchNorm1d

    if settings.DATA_DIMENSIONS == 2:
        return torch.nn.BatchNorm2d

    return torch.nn.BatchNorm3d
