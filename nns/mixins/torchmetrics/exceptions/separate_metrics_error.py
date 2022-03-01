# -*- coding: utf-8 -*-
""" nns/mixins/torchmetrics/exceptions/separate_metrics_error """


__all__ = ['SeparateMetricsError']


class SeparateMetricsError(RuntimeError):
    """
    Exception to be raised by TorchMetricsMixin.get_separate_metrics when the provided metrics_list
    does not match the any signature of the get_separate_metrics_<> methods
    """

    def __init__(self):
        """ Initializes the instance with a custome message """
        super().__init__('Logic not implemented for the provided metrics list')
