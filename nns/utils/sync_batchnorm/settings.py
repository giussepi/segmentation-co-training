# -*- coding: utf-8 -*-
""" nns/utils/sync_batchnorm/settings.py """

try:
    import settings
except ModuleNotFoundError:
    settings = None


CUDA = getattr(settings, 'CUDA', True)

MULTIGPUS = getattr(settings, 'MULTIGPUS', True)

PATCH_REPLICATION_CALLBACK = getattr(settings, 'PATCH_REPLICATION_CALLBACK', True)

# Use 1 for 1D data [batch, channel] or [batch, channel, length]
# Use 2 for 2D data [batch, channel, height, width],
# use 3 for 3D data [batch, channel, depth, height, width].
DATA_DIMENSIONS = getattr(settings, 'DATA_DIMENSIONS', 2)
