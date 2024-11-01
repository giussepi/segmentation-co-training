# -*- coding: utf-8 -*-
""" nns/mixins/settings """

try:
    import settings
except ModuleNotFoundError:
    settings = None

USE_AMP = settings.USE_AMP if hasattr(settings, 'USE_AMP') else False
DISABLE_PROGRESS_BAR = settings.DISABLE_PROGRESS_BAR if hasattr(settings, 'DISABLE_PROGRESS_BAR') else False
