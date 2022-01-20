# -*- coding: utf-8 -*-
""" nns/mixins/settings """

try:
    import settings
except ModuleNotFoundError:
    settings = None

USE_AMP = settings.USE_AMP if hasattr(settings, 'USE_AMP') else False
