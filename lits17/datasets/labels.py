# -*- coding: utf-8 -*-
""" lits17/datasets/labels """

from gtorch_utils.datasets.labels import DatasetLabelsMixin, Detail
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['LiTS17Labels', 'LiTS17OnlyLiverLabels', 'LiTS17OnlyLesionLabels']


class LiTS17OnlyLiverLabels(DatasetLabelsMixin):
    """ Holds data associated with the LiTS17 using two labels: Other and Liver """

    OTHER = Detail('black', 0, 'Other', '', (0, 0, 0))
    LIVER = Detail('white', 1, 'Liver', '', (255, 255, 255))
    LABELS = (OTHER, LIVER)
    CMAPS = tuple(
        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
        for label in LABELS
    )


class LiTS17OnlyLesionLabels(DatasetLabelsMixin):
    """ Holds data associated with the LiTS17 labels """

    OTHER = Detail('black', 0, 'Other', '', (0, 0, 0))
    LESION = Detail('white', 1, 'Tumour lesion', '', (255, 255, 255))
    LABELS = (OTHER, LESION)
    CMAPS = tuple(
        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
        for label in LABELS
    )


class LiTS17Labels(DatasetLabelsMixin):
    """ Holds data associated with the LiTS17 labels """

    OTHER = Detail('black', 0, 'Other', '', (0, 0, 0))
    LIVER = Detail('white', 1, 'Liver', '', (255, 255, 255))
    LESION = Detail('crimson', 2, 'Tumour lesion', '', (220, 20, 60))
    LABELS = (OTHER, LIVER, LESION)
    CMAPS = tuple(
        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
        for label in LABELS
    )
