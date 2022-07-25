# -*- coding: utf-8 -*-
""" ct82/datasets/labels """

from gtorch_utils.datasets.labels import DatasetLabelsMixin, Detail
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['CT82Labels']


class CT82Labels(DatasetLabelsMixin):
    """ Holds data associated with the CT-82 labels """

    OTHER = Detail('black', 0, 'Other', '', (0, 0, 0))
    PANCREAS = Detail('white', 1, 'Pancreas', '', (255, 255, 225))
    LABELS = (OTHER, PANCREAS)
    CMAPS = tuple(
        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
        for label in LABELS
    )
