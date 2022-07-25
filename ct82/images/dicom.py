# -*- coding: utf-8 -*-
""" ct82/images/dicom """

import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from gutils.files import get_filename_and_extension
from gutils.numpy_.numpy_ import scale_using_general_min_max_values
from PIL import Image
from scipy.ndimage import zoom
from skimage.exposure import equalize_hist, equalize_adapthist

from ct82.constants import DICOM_MAX_VAL, DICOM_MIN_VAL


__all__ = ['DICOM']


class DICOM:
    """
    Handles basic operations over DICOM images
    """

    DICOM_EXTENSION = 'dcm'

    def __init__(self, file_path: str):
        """
        Loads the data and initializes the object instance
        """
        assert os.path.isfile(file_path), f'{file_path} does not exists'

        self.file_path = file_path
        self.ndarray = dicom.dcmread(file_path).pixel_array.astype(np.int16)

    def equalize_histogram(self, *, clahe: bool = False, saving_path: str = ''):
        """
        Applies histogram equalization or CLAHE and returns the result as a np.ndarray.

        Kwargs:
            clahe   <bool>: Whether or not apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
                            Default False
            saving_path <str>: full path to save the image. If not provided the equalized image is not
                            saved. Default ''

        Returns:
            equalized_image <np.ndarray>
        """
        assert isinstance(clahe, bool), type(clahe)
        assert isinstance(saving_path, str), type(saving_path)

        rescaled_img = scale_using_general_min_max_values(self.ndarray.astype(
            float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 255))
        img = np.asarray(Image.fromarray(rescaled_img).convert("L"))
        img = equalize_adapthist(img) if clahe else equalize_hist(img)
        img *= 255
        img = np.uint8(img)

        if saving_path:
            Image.fromarray(img).convert("L").save(saving_path)

        return img

    def plot(self):
        plt.imshow(self.ndarray)
        plt.show()

    def resize(self, target: tuple, *, inplace: bool = False):
        """
        Kwargs:
            target <tuple>: Target dimensions. E.g. (256, 256)
            inplace <bool>: Whether or not perform the resize operation in the original array. Default False

        Returns:
            resized <np.ndarray>
        """
        assert isinstance(target, tuple), type(target)
        assert len(self.shape) == len(target), \
            'The shapes of ndarray and targets must have the same number of elements'
        assert isinstance(inplace, bool), type(inplace)

        zoom_factors = np.array(target) / np.array(self.shape)
        resized = zoom(self.ndarray, zoom_factors)
        assert resized.shape == target, 'resize results do not match the target dimensions'

        if inplace:
            self.ndarray = resized

        return resized

    def save_as(self, file_path: str = 'new_image.png', /, *, gray_scale: bool = True):
        """
        Saves the object as a new dcm, jpg, png, tiff, bmp, etc

        Kwargs:
            file_path  <str>: full path to save the image
            gray_scale <boo>: Whether or not save the image in gray scale model.
                              Only applies for non DICOM images.
                              Default True
        """
        assert isinstance(file_path, str), type(file_path)
        assert isinstance(gray_scale, bool), type(gray_scale)

        _, extension = get_filename_and_extension(file_path)

        if extension == self.DICOM_EXTENSION:
            dicom_obj = dicom.dcmread(self.file_path)
            dicom_obj.PixelData = self.ndarray.astype(np.int16).tobytes()
            dicom_obj.Rows, dicom_obj.Columns = self.shape
            dicom_obj.save_as(file_path)
        else:
            rescaled = scale_using_general_min_max_values(self.ndarray.astype(
                float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 255))
            img = Image.fromarray(rescaled)

            if gray_scale and img.mode != 'L':
                img = img.convert("L")
            else:
                img = img.convert('RGB')

            img.save(file_path)

    @property
    def shape(self):
        return self.ndarray.shape
