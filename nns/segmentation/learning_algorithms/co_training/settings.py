# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/settings.py """

try:
    import settings
except ModuleNotFoundError:
    settings = None

DISABLE_PROGRESS_BAR = settings.DISABLE_PROGRESS_BAR if hasattr(settings, 'DISABLE_PROGRESS_BAR') else False
