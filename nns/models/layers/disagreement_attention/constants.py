# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/constants.py """


__all__ = ['AttentionMergingType']


class AttentionMergingType:
    """ Holds the options for the implemented attention merging strategies """

    SUM = 'sum'
    MAX = 'max'
    HADAMARD = 'hadamard'

    OPTIONS = (SUM, MAX, HADAMARD)

    @classmethod
    def validate(cls, type_: str):
        """ validates that the provided type is between the implemented options """
        assert type_ in cls.OPTIONS, (
            f'{type_} is not between the valid options SUM: {cls.SUM}, MAX_POOLING: {cls.MAX}, '
            f'HADAMARD: {cls.HADAMARD}'
        )
