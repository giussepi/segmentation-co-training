# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/managers """

from nns.segmentation.learning_algorithms.co_training.base_managers import BaseCoTraining

__all__ = ['CoTraining']


class CoTraining(BaseCoTraining):
    """

    Usage:
        cot = CoTraining(...)
    """
