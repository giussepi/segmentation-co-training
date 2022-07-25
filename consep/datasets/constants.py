# -*- coding: utf-8 -*-
""" consep/datasets/constants """

from gtorch_utils.datasets.labels import DatasetLabelsMixin, Detail
from matplotlib.colors import LinearSegmentedColormap


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


class BinaryCoNSeP(DatasetLabelsMixin):
    """ Holds data associated with the CoNSeP binary labels """

    OTHER = Detail('black', 0, 'Other', '', (0, 0, 0))
    NUCLEI = Detail('white', 1, 'Nuclei', '', (255, 255, 225))
    LABELS = (OTHER, NUCLEI)
    CMAPS = tuple(
        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
        for label in LABELS
    )


# class MulticlassCoNSeP:
#     """  """
