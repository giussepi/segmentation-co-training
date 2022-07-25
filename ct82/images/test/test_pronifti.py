# -*- coding: utf-8 -*-
""" ct82/images/test/test_pronifti """

import os
import unittest

from ct82.constants import TEST_IMAGES_PATH
from ct82.images import ProNIfTI
from ct82.settings import QUICK_TESTS


class Test_ProNIfTI(unittest.TestCase):

    def setUp(self):
        self.new_file = 'new_pronifti.pro.nii.gz'
        self.files = [
            os.path.join(TEST_IMAGES_PATH, 'PANCREAS_0001',
                         '11-24-2015-PANCREAS0001-Pancreas-18957', 'Pancreas-99667', '1-001.dcm'),
            os.path.join(TEST_IMAGES_PATH, 'PANCREAS_0001',
                         '11-24-2015-PANCREAS0001-Pancreas-18957', 'Pancreas-99667', '1-002.dcm')
        ]

    def test_create_save(self):
        ProNIfTI.create_save(self.files, saving_path=self.new_file)
        data = ProNIfTI(self.new_file)
        self.assertEqual(data.shape, (512, 512, 2), data.shape)
        os.remove(self.new_file)

    def test_create_save_different_saving_path(self):
        other_saving_path = 'xuxuca.pro.nii.gz'
        ProNIfTI.create_save(self.files, saving_path=other_saving_path)
        data = ProNIfTI(other_saving_path)
        self.assertEqual(data.shape, (512, 512, 2), data.shape)
        os.remove(other_saving_path)

    def test_create_save_resizing(self):
        ProNIfTI.create_save(
            self.files, processing={'resize': {'target': (256, 256)}}, saving_path=self.new_file)
        data = ProNIfTI(self.new_file)
        self.assertEqual(data.shape, (256, 256, 2), data.shape)
        os.remove(self.new_file)

    @unittest.skipIf(QUICK_TESTS, 'QUICK_TESTS is set to True')
    def test_plot(self):
        ProNIfTI.create_save(
            self.files, processing={'resize': {'target': (256, 256)}}, saving_path=self.new_file)
        data = ProNIfTI(self.new_file)
        data.plot()
        data.plot(1, 1)
        data.plot(2, 3)
        os.remove(self.new_file)


if __name__ == '__main__':
    unittest.main()
