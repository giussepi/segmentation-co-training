# -*- coding: utf-8 -*-
""" ct82/settings """

try:
    import settings as global_settings
except ModuleNotFoundError:
    global_settings = None


QUICK_TESTS = getattr(global_settings, 'QUICK_TESTS', False)
