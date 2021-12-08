# -*- coding: utf-8 -*-
""" consep/datasets/constants """

import os
from collections import namedtuple

from gutils.plot.tables import plot_color_table
from matplotlib.colors import LinearSegmentedColormap


Detail = namedtuple('Detail', ['colour', 'id', 'name', 'file_label', 'RGB'])


class Dataset:
    """ Holds the datasets names """

    KUMAR = 'kumar'
    CoNSeP = 'consep'

    LIST = [KUMAR, CoNSeP]

    @classmethod
    def validate(cls, db):
        """
        Validates that db belongs to the defined datasets

        Args:
            db <str>: db to validate
        """
        assert isinstance(db, str), type(db)

        if not db in cls.LIST:
            raise Exception(f"Unknown db: {db}")

        return True


class BinaryCoNSeP:
    """ Holds data associated with the CoNSeP binary labels """

    OTHER = Detail('gold', 0, 'Other', '', (244, 220, 5))
    NUCLEI = Detail('royalblue', 1, 'Nuclei', '', (80, 94, 225))
    LABELS = (OTHER, NUCLEI)
    CMAPS = tuple(
        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
        for label in LABELS
    )

    # TODO: this method can be reused if set on an abstract Label class
    @classmethod
    def plot_palette(cls, saving_path=''):
        """
        Plots the colour palette and returns a matplotlib Figure.
        If a saving_path is provided, the palette is saved

        Args:
            saving_path <str>: path to the file to save the image. If not provided the palette
                               will not be saved. Default ''

        Usage:
            import matplotlib.pyplot as plt
            fig = BinaryCoNSeP.plot_palette('<path to my director>my_palettet.png')
            plt.show()

        Returns:
            figure <matplotlib.figure.Figure>
        """
        assert isinstance(saving_path, str), type(saving_path)

        label_colours = {}

        for label in cls.LABELS:
            label_colours[label.name] = [c/255 for c in label.RGB]

        fig = plot_color_table(label_colours, "Label colours")

        if saving_path:
            dirname = os.path.dirname(saving_path)

            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname)

            fig.savefig(saving_path)

        return fig


# class MulticlassCoNSeP:
#     """  """
