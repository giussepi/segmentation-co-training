# -*- coding: utf-8 -*-
""" nns/utils/sync_batchnorm/utils """

import torch

import settings
from .batchnorm import SynchronizedBatchNorm2d


def get_batchnorm2d_class():
    """
    Returns the right BatchNorm2d based on the number of GPUS

    Returns:
        torch.nn.BatchNorm2d or SynchronizedBatchNorm2d
    """
    # TODO: decide if the value from settings should be parameters or just imported
    # from the settings module (like it is working right now)
    if settings.CUDA == settings.MULTIGPUS == settings.PATCH_REPLICATION_CALLBACK == True and \
       torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return SynchronizedBatchNorm2d

    return torch.nn.BatchNorm2d
