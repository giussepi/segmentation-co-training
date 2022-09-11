# -*- coding: utf-8 -*-
""" ct82/processors/test/test_ct82mgr """

import glob
import os
import shutil
import unittest
from unittest.mock import patch

from gutils.images.images import NIfTI, ProNIfTI
from gutils.mock import notqdm

from ct82.processors import CT82MGR
from ct82.constants import TEST_DATASET_PATH


class Test_CT82MGR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.target_size = (160, 160, 96)
        cls.mgr = CT82MGR(
            db_path=TEST_DATASET_PATH,
            cts_path='images',
            labels_path='labels',
            saving_path='test_CT-82-pro',
            target_size=cls.target_size
        )
        cls.mgr.non_existing_ct_folders = []
        cls.mgr()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.mgr.saving_path)

    def test_process(self):
        self.assertEqual(len(glob.glob(os.path.join(self.mgr.saving_labels_folder, r'*.nii.gz'))), 2)
        self.assertEqual(len(glob.glob(os.path.join(self.mgr.saving_cts_folder, r'*.pro.nii.gz'))), 2)

        for subject in range(1, 3):
            labels = NIfTI(os.path.join(self.mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
            cts = ProNIfTI(os.path.join(self.mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
            self.assertTrue(labels.shape == cts.shape == self.target_size)

    def test_perform_visual_verification(self):
        self.mgr.perform_visual_verification(1, scans=2, clahe=True)
        # mgr.perform_visual_verification(1, scans=[72], clahe=True)
        self.assertTrue(os.path.isfile(self.mgr.VERIFICATION_IMG))
        os.remove(self.mgr.VERIFICATION_IMG)

    @patch('ct82.processors.ct82mgr.tqdm', notqdm)
    def test_get_insights(self):
        min_, max_, dicoms_analized, min_slices, max_slices, nifties_analyzed, min_dicoms, max_dicoms = \
            self.mgr.get_insights()

        self.assertEqual(dicoms_analized, 435, dicoms_analized)
        self.assertTrue(min_ >= -1024, min_)
        self.assertTrue(max_ <= 2421, max_)
        self.assertTrue(min_slices >= 46, min_slices)
        self.assertTrue(max_slices <= 145, max_slices)
        self.assertEqual(nifties_analyzed, 2, nifties_analyzed)
        self.assertEqual(min_dicoms >= 181, min_dicoms)
        self.assertTrue(max_dicoms <= 466, max_dicoms)


class Test_CT82MGR_(unittest.TestCase):
    """
    Similar to Test_CT82MGR but now using a target_size (160, 160, -1)
    """

    @classmethod
    def setUpClass(cls):
        cls.target_size = (160, 160, -1)
        cls.mgr = CT82MGR(
            db_path=TEST_DATASET_PATH,
            cts_path='images',
            labels_path='labels',
            saving_path='test_CT-82-pro',
            target_size=cls.target_size
        )
        cls.mgr.non_existing_ct_folders = []
        cls.mgr()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.mgr.saving_path)

    def test_process(self):
        self.assertEqual(len(glob.glob(os.path.join(self.mgr.saving_labels_folder, r'*.nii.gz'))), 2)
        self.assertEqual(len(glob.glob(os.path.join(self.mgr.saving_cts_folder, r'*.pro.nii.gz'))), 2)

        shapes = []
        for subject in range(1, 3):
            labels = NIfTI(os.path.join(self.mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
            cts = ProNIfTI(os.path.join(self.mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
            self.assertTrue(labels.shape == cts.shape)
            self.assertTrue(labels.shape != self.target_size)
            self.assertEqual(labels.shape[:2], self.target_size[:2])
            shapes.append(labels.shape)

        self.assertEqual(shapes[0][:2], shapes[1][:2])
        self.assertNotEqual(shapes[0], shapes[1])

    def test_perform_visual_verification(self):
        self.mgr.perform_visual_verification(1, scans=2, clahe=True)
        # mgr.perform_visual_verification(1, scans=[72], clahe=True)
        self.assertTrue(os.path.isfile(self.mgr.VERIFICATION_IMG))
        os.remove(self.mgr.VERIFICATION_IMG)

    @patch('ct82.processors.ct82mgr.tqdm', notqdm)
    def test_get_insights(self):
        min_, max_, dicoms_analized, min_slices, max_slices, nifties_analyzed, min_dicoms, max_dicoms = \
            self.mgr.get_insights()

        self.assertEqual(dicoms_analized, 435, dicoms_analized)
        self.assertTrue(min_ >= -1024, min_)
        self.assertTrue(max_ <= 2421, max_)
        self.assertTrue(min_slices >= 46, min_slices)
        self.assertTrue(max_slices <= 145, max_slices)
        self.assertEqual(nifties_analyzed, 2, nifties_analyzed)
        self.assertEqual(min_dicoms >= 181, min_dicoms)
        self.assertTrue(max_dicoms <= 466, max_dicoms)


if __name__ == '__main__':
    unittest.main()
