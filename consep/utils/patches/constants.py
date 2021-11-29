# -*- coding: utf-8 -*-
""" consep/utils/patches/constants """


class PatchExtractType:
    """ Holds patch extraction types """

    MIRROR = 'mirror'  # use padding at borders
    VALID = 'valid'  # only extract from valid regions.

    TYPES = [MIRROR, VALID]

    @classmethod
    def validate(cls, patch_type):
        """
        Validates that patch_type belongs to the defined extract types

        Args:
            patch_type <str>: extract type to validate
        """
        assert isinstance(patch_type, str), type(patch_type)

        if not patch_type in cls.TYPES:
            raise Exception(f"Unknonw path type {patch_type}")

        return True
