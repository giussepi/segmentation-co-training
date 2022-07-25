# -*- coding: utf-8 -*-
""" ct82/images/test/test_nifti """

import os
import unittest

import numpy as np

from ct82.constants import TEST_MASKS_PATH
from ct82.images import NIfTI


class Test_NIfTI(unittest.TestCase):

    def setUp(self):
        self.fist_cleaned_slice_idx = 99
        self.slices_with_data = 71
        self.path = os.path.join(TEST_MASKS_PATH, 'label0001.nii.gz')
        self.n = NIfTI(self.path, np.int16)
        self.original_num_slices = self.n.shape[2]

    def test_clean_3d_ndarray(self):
        arr, idx = self.n.clean_3d_ndarray(inplace=False)
        self.assertEqual(arr.shape[2], self.slices_with_data)
        self.assertNotEqual(arr.shape, self.n.shape)
        self.assertEqual(arr.sum(0).sum(0).astype(bool).sum(), self.slices_with_data)
        self.assertEqual(idx.shape[0], self.original_num_slices)
        self.assertEqual(
            idx[self.fist_cleaned_slice_idx:self.fist_cleaned_slice_idx + self.slices_with_data].sum(),
            self.slices_with_data)
        self.assertEqual(idx.sum(), self.slices_with_data)

    def test_clean_3d_ndarray_with_height(self):
        arr, idx = self.n.clean_3d_ndarray(height=55, inplace=False)
        self.assertEqual(arr.shape[2], 55)
        self.assertNotEqual(arr.shape, self.n.shape)
        self.assertEqual(arr.sum(0).sum(0).astype(bool).sum(), 55)
        self.assertEqual(idx.shape[0], self.original_num_slices)
        self.assertEqual(idx[self.fist_cleaned_slice_idx:self.fist_cleaned_slice_idx+55].sum(), 55)
        self.assertEqual(idx.sum(), 55)

    def test_clean_3d_ndarraywith_height_bigger_than_sliced_with_masks_1(self):
        arr, idx = self.n.clean_3d_ndarray(height=100, inplace=False)
        self.assertEqual(arr.shape[2], 100)
        self.assertNotEqual(arr.shape, self.n.shape)
        self.assertEqual(arr.sum(0).sum(0).astype(bool).sum(), self.slices_with_data)
        self.assertEqual(idx.shape[0], self.original_num_slices)
        self.assertEqual(idx[self.fist_cleaned_slice_idx:self.fist_cleaned_slice_idx+100].sum(), 100)
        self.assertEqual(idx.sum(), 100)

    def test_clean_3d_ndarray_with_height_bigger_than_slices_with_masks_2(self):
        arr, idx = self.n.clean_3d_ndarray(height=230, inplace=False)
        self.assertEqual(arr.shape[2], 230)
        self.assertNotEqual(arr.shape, self.n.shape)
        self.assertEqual(arr.sum(0).sum(0).astype(bool).sum(), self.slices_with_data)
        self.assertEqual(idx.shape[0], self.original_num_slices)
        self.assertEqual(idx[-230:].sum(), 230)
        self.assertEqual(idx.sum(), 230)

    def test_clean_3d_ndarray_height_bigger_than_slices(self):
        with self.assertRaises(AssertionError):
            arr, idx = self.n.clean_3d_ndarray(height=241, inplace=False)

    def test_clean_3d_ndarray_inplace(self):
        arr, idx = self.n.clean_3d_ndarray(inplace=True)
        self.assertEqual(self.n.shape[2], self.slices_with_data)
        self.assertEqual(id(arr), id(self.n.ndarray))
        self.assertEqual(self.n.ndarray.sum(0).sum(0).astype(bool).sum(), self.slices_with_data)
        self.assertEqual(idx.shape[0], self.original_num_slices)
        self.assertNotEqual(idx.shape[0], self.n.shape[2])
        self.assertEqual(
            idx[self.fist_cleaned_slice_idx:self.fist_cleaned_slice_idx + self.slices_with_data].sum(),
            self.slices_with_data)
        self.assertEqual(idx.sum(), self.slices_with_data)


if __name__ == '__main__':
    unittest.main()
