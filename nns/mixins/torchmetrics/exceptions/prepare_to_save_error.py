# -*- coding: utf-8 -*-
""" nns/mixins/torchmetrics/exceptions/prepare_to_save_error """


__all__ = ['PrepareToSaveError']


class PrepareToSaveError(RuntimeError):
    """
    Exception to be raised by TorchMetricsMixin.prepare_to_save when the provided data does
    not match the signature of any of the prepare_to_save methods.
    """

    def __init__(self):
        """ Initializes the instance with a custome message """
        super().__init__('Tensor cleaning logic not implemented for the provided data')
