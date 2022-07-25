# -*- coding: utf-8 -*-
""" ct82/constants """

import os

DICOM_MIN_VAL = -2048
DICOM_MAX_VAL = 3071

TEST_DATASET_PATH = os.path.join('ct82', 'test_datasets', 'CT-82')
TEST_IMAGES_PATH = os.path.join(TEST_DATASET_PATH, 'images')
TEST_MASKS_PATH = os.path.join(TEST_DATASET_PATH, 'labels')
