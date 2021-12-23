# -*- coding: utf-8 -*-
""" nns/callbacks/metrics/constants """


class MetricEvaluatorMode:
    """
    Holds the options for nns.callbacks.metrics.metric_evaluator.MetricEvaluator
    """
    MIN = 0  # minize the metric
    MAX = 1  # maximize the metric

    OPTIONS = (MIN, MAX)

    @classmethod
    def validate(cls, opt):
        """
        Validates the option

        Kwargs:
            opt <int>: option
        """
        assert opt in cls.OPTIONS, f'{opt} is not in {cls.__name__}.OPTIONS'
