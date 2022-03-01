# -*- coding: utf-8 -*-
""" nns/mixins/exceptions/model_mgr """


__all__ = ['IniCheckpintError']


class IniCheckpintError(RuntimeError):
    """
    Exception to be raised by descendants of ModelMGRMixin when the ini_checkpoint
    argument is not provided and it is necessary
    """

    def __init__(self):
        """ Initializes the instance with a custome message """
        super().__init__('The ModelMGR instance does not have a valid ini_checkpoint attribute')
