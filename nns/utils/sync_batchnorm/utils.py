# -*- coding: utf-8 -*-
""" nns/utils/sync_batchnorm/utils """

import torch

import settings
from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d


def get_batchnormxd_class(data_dimensions: int = 2, /):
    """
    Returns the right BatchNorm based on the number of GPUS

    Returns:
        torch.nn.BatchNorm<1|2|3>d or SynchronizedBatchNorm<1|2|3>d

    Args:
        data_dimensions  <int>: Number of dimensions of the data.
                                Use 1 for 1D data [batch, channel] or [batch, channel, length]
                                Use 2 for 2D data [batch, channel, height, width],
                                use 3 for 3D data [batch, channel, depth, height, width].
                                Default 2
    """
    assert data_dimensions in (1, 2, 3), 'Only 1D, 2D and 3D data is supported'

    # TODO: decide if the value from settings should be parameters or just imported
    # from the settings module (like it is working right now)
    if settings.CUDA == settings.MULTIGPUS == settings.PATCH_REPLICATION_CALLBACK == True and \
       torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if data_dimensions == 1:
            return SynchronizedBatchNorm1d

        if data_dimensions == 2:
            return SynchronizedBatchNorm2d

        return SynchronizedBatchNorm3d

    if data_dimensions == 1:
        return torch.nn.BatchNorm1d

    if data_dimensions == 2:
        return torch.nn.BatchNorm2d

    return torch.nn.BatchNorm3d
