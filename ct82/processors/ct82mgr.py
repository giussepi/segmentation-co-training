# -*- coding: utf-8 -*-
""" ct82/processors/ct82mgr """

import glob
import os
import re
import shutil
import time
from typing import Union, Optional

import numpy as np
from gtorch_utils.constants import DB
from gutils.datasets.utils.split import TrainValTestSplit
from gutils.decorators import timing
from gutils.files import get_filename_and_extension
from gutils.folders import clean_create_folder
from gutils.numpy_.numpy_ import scale_using_general_min_max_values
from PIL import Image
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from ct82.constants import DICOM_MIN_VAL, DICOM_MAX_VAL
from ct82.images import DICOM, NIfTI, ProNIfTI


__all__ = ['CT82MGR']


class CT82MGR:
    """
    Contains all the logic to methods to process the CT-82 and a dataset that can be consumed by a
    model

    Usage:
        CT82MGR()()
    """

    SAVING_LABELS_FOLDER = 'labels'
    SAVING_CTS_FOLDER = 'images'
    GEN_LABEL_FILENAME_TPL = 'label_{:02d}.nii.gz'
    GEN_CT_FILENAME_TPL = 'CT_{:02}.pro.nii.gz'
    LABEL_REGEX = r'.+label[0]*(?P<id>\d+).nii.gz'
    VERIFICATION_IMG = 'visual_verification.png'

    def __init__(
            self,
            db_path: str = 'CT-82',
            cts_path: str = os.path.join('manifest-1599750808610', 'Pancreas-CT'),
            labels_path: str = 'TCIA_pancreas_labels-02-05-2017',
            saving_path: str = 'CT-82-Pro',
            target_size: tuple = None,
            non_existing_ct_folders: tuple = None
    ):
        """
        Initialized the object instance

        Kwargs:
            db_path     <str>: Path to the main folder containing the dataset. Default 'CT-82'
            cts_path    <str>: path (relative to db_path) to the main folder containing the CT scans.
                               Default 'manifest-1599750808610/Pancreas-CT'
            labels_path <str>: path (relative to db_path) to the main folder containing the labels/masks.
                               Default 'TCIA_pancreas_labels-02-05-2017'
            saving_path <str>: Path to the folder where the processed labels and CTs will be stored
                               Default CT-82-Pro
            target_size <tuple>: target size of the 3D data height x width x slices.
                               Default (368, 368, 96)
            non_existing_ct_folders <tuple>: Tuple containing the id of the non existing CT folders.
                               Default [25, 70].
                               For the Pancreas CT-82 version 2020/09/10 the cases 25 and 70 were
                               found to be from the same scan #2, just cropped slightly differently,
                               so they were removed
                               https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#225140409ddfdf8d3b134d30a5287169935068e3
        """
        self.db_path = db_path
        self.cts_path = os.path.join(self.db_path, cts_path)
        self.labels_path = os.path.join(self.db_path, labels_path)
        self.saving_path = saving_path
        self.target_size = target_size if target_size is not None else (368, 368, 96)
        self.non_existing_ct_folders = \
            non_existing_ct_folders if non_existing_ct_folders is not None else (25, 70)

        assert os.path.isdir(self.db_path), self.db_path
        assert os.path.isdir(self.cts_path), self.cts_path
        assert os.path.isdir(self.labels_path), self.labels_path
        assert isinstance(self.saving_path, str), type(self.saving_path)
        assert isinstance(self.target_size, tuple), type(tuple)
        assert len(self.target_size) == 3, len(self.target_size)
        assert isinstance(self.non_existing_ct_folders, tuple), type(self.non_existing_ct_folders)

        self.saving_labels_folder = os.path.join(self.saving_path, self.SAVING_LABELS_FOLDER)
        self.saving_cts_folder = os.path.join(self.saving_path, self.SAVING_CTS_FOLDER,)
        self.label_pattern = re.compile(self.LABEL_REGEX)

    def __call__(self):
        self.process()

    def _process_labels_cts(self, labels_file: str, cts_folder: str, /):
        """
        Processes the NIfTI labels and DICOM scans provided, then results are saved two NIfTI files

        Kwargs:
            labels_file <str>: path to the NIfTI file to be processed
            cts_folder  <str>: path to the folder containing the DICOM scans corresponding to the
                               provided labels_file
        """
        labels = NIfTI(labels_file)
        _, selected_data_idx = labels.clean_3d_ndarray(height=self.target_size[2], inplace=True)
        # then we apply the resize over the annotated area (ROI)
        labels.resize(self.target_size, inplace=True)
        result = self.label_pattern.match(labels_file)

        if result is None:
            raise RuntimeError(f'The label id from {labels_file} could not be retrieved.')

        label_id = int(result.groupdict()['id'])
        labels.save_as(os.path.join(self.saving_labels_folder, self.GEN_LABEL_FILENAME_TPL.format(label_id)))
        cts_wildcard = os.path.join(cts_folder, '**/*.dcm')
        selected_dicoms = \
            np.array(sorted(glob.glob(cts_wildcard, recursive=True)))[selected_data_idx].tolist()
        ProNIfTI.create_save(
            selected_dicoms,
            processing={'resize': {'target': self.target_size[:2]}},
            saving_path=os.path.join(self.saving_cts_folder, self.GEN_CT_FILENAME_TPL.format(label_id))
        )

    @timing
    def process(self):
        """
        Processes all the labels/masks and CT scans and saves the results
        """
        labels_wildcard = os.path.join(self.labels_path, '*.nii.gz')
        labels_files = glob.glob(labels_wildcard)
        labels_files.sort()
        cts_folders = [os.path.join(self.cts_path, f'PANCREAS_{i:04d}') for i in range(1, 83)]
        cts_folders.sort()

        clean_create_folder(self.saving_labels_folder)
        clean_create_folder(self.saving_cts_folder)

        for idx in self.non_existing_ct_folders[::-1]:
            cts_folders.pop(idx-1)
            labels_files.pop(idx-1)

        for labels_file, cts_folder in tqdm(zip(labels_files, cts_folders)):
            self._process_labels_cts(labels_file, cts_folder)

    def get_insights(self):
        """
        Browse all the ct82 DICOM and NIfTI files and returns the min/max DICOM values,
        total DICOMs analized, min/max number of NIfTI slices with data/masks, total NIfTIs analyzed,
        min/max DICOMs per subject

        Returns:
            min_dicom_val<int>, max_dicom_val<int>, dicoms_analized<int>,
            min_slices_with_data<int>, max_slices_with_data<int>, nifties_analyzed<int>,
            min_dicoms_per_subject<int>, max_dicoms_per_subject
        """
        cts_wildcard = os.path.join(self.cts_path, '**/*.dcm')
        dicoms = glob.glob(cts_wildcard, recursive=True)
        dicoms_analized = len(dicoms)
        min_dicom_val = min_slices_with_data = min_dicoms_per_subject = np.inf
        max_dicom_val = max_slices_with_data = max_dicoms_per_subject = np.NINF

        for dicom_path in tqdm(dicoms, unit='DICOMs'):
            dcm = DICOM(dicom_path)
            min_dicom_val = min(min_dicom_val, dcm.ndarray.min())
            max_dicom_val = max(max_dicom_val, dcm.ndarray.max())

        nifti_wildcard = os.path.join(self.labels_path, '*.nii.gz')
        niftis = glob.glob(nifti_wildcard)
        niftis_analyzed = len(niftis)

        for nifti_path in tqdm(glob.glob(nifti_wildcard), unit="NIfTIs"):
            nifti = NIfTI(nifti_path)
            nifti.clean_3d_ndarray(inplace=True)
            min_slices_with_data = min(min_slices_with_data, nifti.shape[2])
            max_slices_with_data = max(max_slices_with_data, nifti.shape[2])

        # heree
        cts_folders = [os.path.join(self.cts_path, f'PANCREAS_{i:04d}') for i in range(1, 83)]
        cts_folders.sort()

        for idx in self.non_existing_ct_folders[::-1]:
            cts_folders.pop(idx-1)

        for subject_dicoms_folder in tqdm(cts_folders, unit='DICOMs'):
            subject_dicoms = len(glob.glob(os.path.join(subject_dicoms_folder, '**/*.dcm'), recursive=True))
            min_dicoms_per_subject = min(min_dicoms_per_subject, subject_dicoms)
            max_dicoms_per_subject = max(max_dicoms_per_subject, subject_dicoms)

        return min_dicom_val, max_dicom_val, dicoms_analized, min_slices_with_data, max_slices_with_data, \
            niftis_analyzed, min_dicoms_per_subject, max_dicoms_per_subject

    def perform_visual_verification(
            self, subject_id: int, /, *, alpha: float = .4, scans: Union[int, list] = 1, clahe: bool = False,
            rgb_mask: tuple = None
    ):
        """
        To see the changes open visual_verification.png and it will be continuosly updated with new mask data

        Kwargs:
            subject_id  <int>: chosen id, without leading zeros, from the generated labels (label_<id>.nii.gz)
                               or CTs (CT_<id>.pro.nii.gz) to be analyzed.
            alpha     <float>: alpha channel value in range ]0, 1]. Default 0.9
            scans <int, list>: If set to an integer it will be the number of scans to be analyzed. If
                               a single-element list is provided it will be the number of scans
                               to be analyzed, if a two-element list is provided it will be the interval
                               of slices to be analyzed. Default 1
            clahe   <bool>: Whether or not apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
                            Default False
            rgb_mask   <bool>: RGB colour for the mask. Default (148, 36, 36)
        """
        rgb_mask = rgb_mask if rgb_mask else (148, 36, 36)

        assert isinstance(subject_id, int), type(subject_id)
        assert 0 < alpha <= 1, alpha
        assert isinstance(scans, (int, list)), type(scans)
        if isinstance(scans, list):
            assert len(scans) <= 2, 'scan list can have two elements at most'
            if len(scans) == 2:
                assert scans[0] < scans[1]
        assert isinstance(clahe, bool), type(clahe)
        assert isinstance(rgb_mask, tuple), type(rgb_mask)

        label = NIfTI(os.path.join(self.saving_labels_folder, self.GEN_LABEL_FILENAME_TPL.format(subject_id)))
        ct = ProNIfTI(os.path.join(self.saving_cts_folder, self.GEN_CT_FILENAME_TPL.format(subject_id)))

        scans = [scans] if isinstance(scans, int) else [*range(*scans)]

        for scan in scans:
            mask = label.ndarray[..., scan] * 255
            image = scale_using_general_min_max_values(ct.ndarray[..., scan].astype(
                float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 255))

            if clahe:
                image = np.asarray(Image.fromarray(np.uint8(image)).convert('L'))
                image = equalize_adapthist(image)*255
                image = image.astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image.astype(np.uint8)).convert('RGB')

            mask = Image.fromarray(mask.T.astype(np.uint8)).convert('RGBA')
            new_data = []

            # setting black as transparent
            for item in mask.getdata():
                if item[0] == item[1] == item[2] == 0:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append((*rgb_mask, 255))

            mask.putdata(new_data)
            mask_trans = Image.new("RGBA", mask.size)
            mask_trans = Image.blend(mask_trans, mask, alpha)
            image.save(self.VERIFICATION_IMG)
            time.sleep(1)
            image.paste(mask_trans, (0, 0), mask_trans)
            image.save(self.VERIFICATION_IMG)
            time.sleep(1)
            # plt.imshow(np.asarray(image))
            # plt.pause(1)
            # plt.close()

    def split_processed_dataset(
            self, val_size: float = .1, test_size: float = .2, /, *, random_state: int = 42,
            shuffle: bool = True
    ):
        """
        Splits the processet CT-82 into train, validation and test subdatasets

        Kwargs:
            val_size  <float>: validation dataset size in range [0, 1].
                               Default .1
            test_size <float>: test dataset size in range [0, 1].
                               Default .2
            random_state <int>: Controls the shuffling applied to the data before applying the split.
                               Default 42
            shuffle    <bool>: Whether or not to shuffle the data before splitting.
                               Default True
        """
        assert 0 <= val_size < 1, val_size
        assert 0 <= test_size < 1, test_size
        assert val_size + test_size < 1, (val_size, test_size)
        assert isinstance(random_state, int), type(int)
        assert isinstance(shuffle, bool), type(bool)

        labels_wildcard = os.path.join(self.saving_labels_folder, '*.nii.gz')
        labels_files = glob.glob(labels_wildcard)
        labels_files.sort()
        cts_wildcard = os.path.join(self.saving_cts_folder, '*.pro.nii.gz')
        cts_files = glob.glob(cts_wildcard)
        cts_files.sort()
        destination_folders = [
            os.path.join(self.saving_path, DB.TRAIN),
            os.path.join(self.saving_path, DB.VALIDATION),
            os.path.join(self.saving_path, DB.TEST)
        ]

        x_train, x_val, x_test, y_train, y_val, y_test = TrainValTestSplit(
            np.array(labels_files), np.array(cts_files), val_size=val_size, test_size=test_size,
            random_state=random_state, shuffle=shuffle
        )()

        for subdataset in destination_folders:
            clean_create_folder(subdataset)

        subdatasets = [
            np.concatenate([x_train, y_train]),
            np.concatenate([x_val, y_val]),
            np.concatenate([x_test, y_test])
        ]

        for folder, filepaths in tqdm(zip(destination_folders, subdatasets)):
            for filepath in filepaths:
                shutil.move(
                    filepath,
                    os.path.join(folder, '.'.join(get_filename_and_extension(filepath)))
                )

        os.rmdir(self.saving_labels_folder)
        os.rmdir(self.saving_cts_folder)
