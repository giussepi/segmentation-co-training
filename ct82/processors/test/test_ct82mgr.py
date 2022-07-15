# -*- coding: utf-8 -*-
""" ct82/processors/test/test_ct82mgr """

import glob
import os
import shutil
import unittest

from ct82.images import NIfTI, ProNIfTI
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

    def test_get_min_max_values(self):
        min_, max_, dicoms_analized = self.mgr.get_min_max_values()

        self.assertEqual(dicoms_analized, 435, dicoms_analized)
        self.assertTrue(min_ >= -1024, min_)
        self.assertTrue(max_ <= 2421, max_)


if __name__ == '__main__':
    unittest.main()
