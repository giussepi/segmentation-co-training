# -*- coding: utf-8 -*-
""" ct82/images/test/test_dicom """

import os
import unittest

import numpy as np

from ct82.constants import TEST_IMAGES_PATH
from ct82.images import DICOM


class Test_DICOM(unittest.TestCase):
    def setUp(self):
        self.img = DICOM(os.path.join(TEST_IMAGES_PATH, 'PANCREAS_0001',
                         '11-24-2015-PANCREAS0001-Pancreas-18957', 'Pancreas-99667', '1-001.dcm'))
        self.saving_path = '1-001.png'

    def test_equalize_histogram(self):
        equalized = self.img.equalize_histogram()
        self.assertFalse(np.array_equal(equalized, self.img.ndarray))
        self.assertEqual(equalized.shape, self.img.shape)
        self.assertTrue(0 <= equalized.min() <= 255)
        self.assertTrue(0 <= equalized.max() <= 255)

    def test_equalize_histogram_with_clahe_and_saving_path(self):

        equalized_clahe = self.img.equalize_histogram(clahe=True, saving_path=self.saving_path)
        equalized = self.img.equalize_histogram()
        self.assertFalse(np.array_equal(equalized, equalized_clahe))
        self.assertFalse(np.array_equal(equalized_clahe, self.img.ndarray))
        self.assertEqual(equalized_clahe.shape, self.img.shape)
        self.assertTrue(0 <= equalized_clahe.min() <= 255)
        self.assertTrue(0 <= equalized_clahe.max() <= 255)
        self.assertTrue(os.path.isfile(self.saving_path))
        os.remove(self.saving_path)


if __name__ == '__main__':
    unittest.main()
