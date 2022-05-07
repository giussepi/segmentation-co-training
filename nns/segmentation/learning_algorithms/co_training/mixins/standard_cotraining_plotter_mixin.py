# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/mixins/standard_cotraining_plotter_mixin """

from nns.segmentation.learning_algorithms.co_training.mixins.base_cotraining_plotter_mixin import \
    BaseCotrainingPlotterMixin


__all__ = ['CotrainingPlotterMixin']


class CotrainingPlotterMixin(BaseCotrainingPlotterMixin):
    """
    Contains methods to plot data from class CoTraining

    Usage:
        class CoTraining(CotrainingPlotterMixin, ...):
            ...
    """
