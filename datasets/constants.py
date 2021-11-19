# -*- coding: utf-8 -*-
""" datasets/constants """


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
