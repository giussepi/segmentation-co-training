# -*- coding: utf-8 -*-
""" nns/utils/metrics """

import torchmetrics


class MetricItem:
    """
    Object instance to wrap metrics to be used with the ModelMGR

    Usage:
        ModelMGR(
            metrics=[
                MetricItem(torchmetrics.DiceCoefficient(), main=True),
                MetricItem(torchmetrics.Specificity(), main=True),
                MetricItem(torchmetrics.Recall())
            ],
            ...
        )()
    """

    def __init__(self, metric: torchmetrics.Metric, /, *, main: bool = False):
        """

        Initializes the objec tinstance

        Args:
            metric <torchmetrics.Metric>: Metric instance

        Kwargs:
            main                  <bool>: Set to True to let the ModelMGR know that this metric
                                          must be used during the procedure to find the best model
        """
        assert isinstance(metric, torchmetrics.Metric), type(metric)
        assert isinstance(main, bool), type(main)

        self.metric = metric
        self.main = main
