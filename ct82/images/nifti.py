# -*- coding: utf-8 -*-
""" ct82/images/nifti """

import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scipy.ndimage import zoom


__all__ = ['NIfTI']


class NIfTI:
    """
    Handles basic operations over NIfTi images
    """

    def __init__(self, filepath: str, dtype: np.dtype = np.int16):
        """ Loads and initialzes the object instance """
        self.ndarray = self.meta = None
        self.filepath = filepath
        self.dtype = dtype

        self._load_nifti_img(filepath, dtype)

    def _load_nifti_img(self, filepath: str, dtype: np.dtype = np.int16):
        '''
        NIFTI Image Loader

        Kwargs:
            filepath <tr>: path to the input NIFTI image
            dtype  <type>: dataio type of the nifti numpy array

        Returns:
            img <np.array> meta <dict>
        '''
        assert os.path.isfile(filepath), f'{filepath} does not exist'
        assert isinstance(dtype, type), type(dtype)

        nim = nib.load(filepath)
        self.ndarray = np.array(nim.get_fdata(), dtype=dtype)
        self.ndarray = np.squeeze(self.ndarray)
        self.meta = {'affine': nim.affine,
                     'dim': nim.header['dim'],
                     'pixdim': nim.header['pixdim'],
                     'name': os.path.basename(filepath)
                     }

    def plot_3d_ndarray(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z, x, y = self.ndarray.nonzero()
        ax.scatter(x, y, z, c=z, alpha=1)
        plt.show()

    @property
    def shape(self):
        return self.ndarray.shape

    def clean_3d_ndarray(self, *, height: int = -1, inplace: bool = False):
        """
        Returns only the slices with data
        NOTE: The algorithm assumes that slices with relevant data are contiguous

        Kwargs:
            height   <int>: Desired number of slices with data to be selected.
                            If set to -1, it there will not be a fixed height.
                            Default -1
            inplace <bool>: Whether or not perform the clean operation in the original array. Default False

        Returns:
            cleaned_array <np.ndarray>, selected_data_idx <np.ndarray>
        """
        if height != -1:
            assert isinstance(height, int), type(height)
            assert height <= self.shape[2], \
                'height must be less than or equalt to the total number of NIfTI slices'
        if inplace:
            assert isinstance(inplace, bool), type(inplace)

        selected_data_idx = self.ndarray.sum(axis=0).sum(axis=0).astype(bool)
        # total_cleaned_data_slices = selected_data_idx.sum()
        fist_cleaned_slice_idx = selected_data_idx.argmax()

        if height != -1:
            selected_data_idx = np.zeros_like(selected_data_idx, dtype=bool)

            if fist_cleaned_slice_idx + height >= self.shape[2]:
                slices_with_data = self.ndarray[:, :, -height:]
                selected_data_idx[-height:] = True
            else:
                slices_with_data = self.ndarray[:, :, fist_cleaned_slice_idx:fist_cleaned_slice_idx+height]
                selected_data_idx[fist_cleaned_slice_idx:fist_cleaned_slice_idx+height] = True
        else:
            slices_with_data = self.ndarray[:, :, selected_data_idx]

        if inplace:
            self.ndarray = slices_with_data

        return slices_with_data, selected_data_idx

    def resize(self, target: tuple, *, inplace: bool = False):
        """
        Kwargs:
            target <tuple>: Target dimensions. E.g. (256, 256, 200)
            inplace <bool>: Whether or not perform the resize operation in the original array. Default False

        Returns:
            resized <np.ndarray>
        """
        assert isinstance(target, tuple), type(target)
        assert len(self.shape) == len(target), \
            'The shapes of ndarray and targets must have the same number of elements'

        zoom_factors = np.array(target) / np.array(self.shape)
        resized = zoom(self.ndarray, zoom_factors)
        assert resized.shape == target, 'resize results do not match the target dimensions'

        if inplace:
            self.ndarray = resized

        return resized

    @staticmethod
    def save_numpy_as_nifti(ndarray: np.ndarray, file_path: str = 'new_image.nii.gz'):
        assert isinstance(ndarray, np.ndarray), type(ndarray)
        assert isinstance(file_path, str), type(file_path)

        img = nib.Nifti1Image(ndarray, np.eye(4))
        nib.save(img, file_path)

    def save(self):
        """ Overwrites the file """
        self.save_numpy_as_nifti(self.ndarray, self.filepath)

    def save_as(self, file_path: str = 'new_nifti.nii.gz'):
        self.save_numpy_as_nifti(self.ndarray, file_path)
