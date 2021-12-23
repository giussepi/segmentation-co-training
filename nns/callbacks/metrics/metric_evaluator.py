# -*- coding: utf-8 -*-
""" nns/callbacks/metrics/metric_evaluator """

from nns.callbacks.metrics.constants import MetricEvaluatorMode


class MetricEvaluator:
    """
    Evaluates a new validation metric against the old best metric
    considering if we need to maximize or minimize the metric

    Usage:
        metric_evaluator = MetricEvaluator()

        if metric_evaluator(val_metric, best_metric):
            # do something like saving your model
            # or updating the best metric value (best_metric = val_metric)
    """

    def __init__(self, mode=MetricEvaluatorMode.MAX):
        """ Initializes the object instance """
        MetricEvaluatorMode.validate(mode)
        self.mode = mode

    def __call__(self, val_metric, best_metric):
        return self.process(val_metric, best_metric)

    def process(self, val_metric, best_metric):
        """
        Returns the comparison between the validation_metric (new value to evaluate) and
        the best_metric (old best value to verify) based on the self.mode

        Kwargs:
            val_metric  <float>: new validation metric value
            best_metric <float>: old best metric value

        Returns:
            metric_comparison <bool>
        """

        if self.mode == MetricEvaluatorMode.MAX:
            return val_metric > best_metric

        return val_metric < best_metric
