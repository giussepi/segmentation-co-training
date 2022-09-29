# -*- coding: utf-8 -*-
""" lits17/processors/lits17mgr """

import glob
import os
import re
import shutil
import time
from typing import Union


import numpy as np
from logzero import logger
from gtorch_utils.constants import DB
from gutils.datasets.utils.split import TrainValTestSplit
from gutils.decorators import timing
from gutils.exceptions.common import ExclusiveArguments
from gutils.files import get_filename_and_extension
from gutils.folders import clean_create_folder
from gutils.images.images import NIfTI
from gutils.numpy_.numpy_ import scale_using_general_min_max_values
from gutils.plots import BoxPlot
from PIL import Image
from skimage.exposure import equalize_adapthist
from tabulate import tabulate
from tqdm import tqdm

from lits17.constants import CT_MIN_VAL, CT_MAX_VAL


__all__ = ['LiTS17MGR']


class LiTS17MGR:
    """
    Contains all the logic to methods to process the LiTS17 and create a dataset that can
    be consumed by a model

    Usage:
        LiTS17MGR('<path_to_lits17_files>')()
    """

    GEN_LABEL_FILENAME_TPL = 'label_{:03d}.nii.gz'
    GEN_CT_FILENAME_TPL = 'CT_{:03d}.nii.gz'
    LABEL_REGEX = r'.*segmentation-(?P<id>\d+).nii'
    GENERIC_REGEX = r'(?P<type>\w+)-(?P<id>\d+).nii'
    VERIFICATION_IMG = 'visual_verification.png'

    def __init__(
            self, db_path: str, /, *, saving_path: str = 'LiTS17-Pro', target_size: tuple = None,
            only_liver: bool = False, only_lesion: bool = False):
        """
        Initialized the object instance

        Kwargs:
            db_path     <str>: Path to the main folder containing the dataset.
            saving_path <str>: Path to the folder where the processed labels and CTs will be stored
                               Default LiTS17-Pro
            target_size <tuple>: target size of the 3D data height x width x slices. If the value of the
                               third dimension is -1, the target size per subject will be
                               (height, width, num_slices_with_data). When the value is -2, only the
                               height and width will be scaled; thus, the target size per subject will
                               be (height, width, original_depth).
                               Default (368, 368, 96)
            only_liver <bool>: If True, the masks will have only two labels: Other and liver
            only_lesion <bool>: If True, the masks will have only two labels: Other and lesion
        """
        self.db_path = db_path
        self.saving_path = saving_path
        self.target_size = target_size if target_size is not None else (368, 368, 96)
        self.only_liver = only_liver
        self.only_lesion = only_lesion

        assert os.path.isdir(self.db_path), self.db_path
        assert isinstance(self.saving_path, str), type(self.saving_path)
        assert isinstance(self.target_size, tuple), type(tuple)
        assert len(self.target_size) == 3, len(self.target_size)
        assert self.target_size[2] in (-1, -2) or self.target_size[2] > 0
        assert isinstance(self.only_liver, bool), type(self.only_liver)
        if self.only_lesion and self.only_liver:
            raise ExclusiveArguments(['only_liver', 'only_lesion'])

        self.label_pattern = re.compile(self.LABEL_REGEX)
        self.generic_pattern = re.compile(self.GENERIC_REGEX)

    def __call__(self):
        self.process()

    def _process_labels_cts(self, labels_file: str, /):
        """
        Processes the label and CT associated with 'labels_file', the saves the results
        """
        labels = NIfTI(labels_file)

        if self.target_size[2] != -2:
            _, selected_data_idx = labels.clean_3d_ndarray(height=self.target_size[2], inplace=True)

        # NOTE: we modify the labels after selecting the data because we must
        #       work over continuos data always; otherwise, subsequent
        #       resample operations will not work properly
        if self.only_liver:
            labels.ndarray = (labels.ndarray > 0).astype(np.int)
        elif self.only_lesion:
            labels.ndarray = (labels.ndarray > 1).astype(np.int)

        image = NIfTI(labels_file.replace('segmentation', 'volume'))

        if self.target_size[2] != -2:
            image.ndarray = image.ndarray[..., selected_data_idx]

        # then we apply the resize over the annotated area (ROI)
        if self.target_size[2] in (-1, -2):
            labels.resize((*self.target_size[:2], labels.shape[2]), inplace=True)
            image.resize((*self.target_size[:2], image.shape[2]), inplace=True)
        else:
            labels.resize(self.target_size, inplace=True)
            image.resize(self.target_size, inplace=True)

        result = self.label_pattern.match(labels_file)

        if result is None:
            raise RuntimeError(f'The label id from {labels_file} could not be retrieved.')

        label_id = int(result.groupdict()['id'])
        labels.save_as(os.path.join(self.saving_path, self.GEN_LABEL_FILENAME_TPL.format(label_id)))
        image.save_as(os.path.join(self.saving_path, self.GEN_CT_FILENAME_TPL.format(label_id)))

    @timing
    def process(self):
        """
        Processes all the labels/masks and CT scans and saves the results
        """
        labels_wildcard = os.path.join(self.db_path, 'segmentation-*.nii')
        labels_files = glob.glob(labels_wildcard)

        labels_files.sort()
        clean_create_folder(self.saving_path)

        for labels_file in tqdm(labels_files):
            self._process_labels_cts(labels_file)

    @timing
    def get_insights(self, *, verbose: bool = False):
        """
        Browse all LiTS17 files from self.db_path and returns the min/max image values,
        min/max number of NIfTI files per label with data/masks, files without label 1,
        files without label 2, min/max height, width and depth; total number of CTs and
        segmentation files analized

        Kwargs:
            verbose <bool>: If True prints all data found. Default False

        Returns:
            min_image_val<int>, max_image_val<int>,
            min_slices_with_label_1<int>, max_slices_with_label_1<int>,
            min_slices_with_label_2<int>, max_slices_with_label_2<int>,
            files_without_label_1<List[int]>, files_wihout_label_2<List[int]>,
            min_height<int>, min_width<int>, min_depth,
            max_height<int>, max_width<int>, max_depth,
            img_counter<int>, label_counter<int>
        """
        wildcard = os.path.join(self.db_path, '*.nii')
        files_ = glob.glob(wildcard)
        min_image_val = min_slices_with_label_1 = min_slices_with_label_2 = \
            min_width = min_height = min_depth = float('inf')
        max_image_val = max_slices_with_label_1 = max_slices_with_label_2 = \
            max_width = max_height = max_depth = float('-inf')
        files_without_label_1 = []
        files_without_label_2 = []
        img_counter = label_counter = 0

        for file_ in tqdm(files_, unit='NIfTI files'):
            filename = os.path.basename(file_)
            result = self.generic_pattern.match(filename)

            if not result:
                logger.warning('%s not analyzed because it does not match the generic '
                               'file pattern %s', filename, self.GENERIC_REGEX)
                continue

            nifti = NIfTI(file_)
            min_height = min(min_height, nifti.shape[0])
            min_width = min(min_width, nifti.shape[1])
            min_depth = min(min_depth, nifti.shape[2])
            max_height = max(max_height, nifti.shape[0])
            max_width = max(max_width, nifti.shape[1])
            max_depth = max(max_depth, nifti.shape[2])

            if result.groupdict()['type'] == 'volume':
                min_image_val = min(min_image_val, nifti.ndarray.min())
                max_image_val = max(max_image_val, nifti.ndarray.max())
                img_counter += 1
            else:
                slices_with_label_1 = \
                    (nifti.ndarray == 1).astype(np.int).sum(axis=0).sum(axis=0).astype(np.bool).sum()
                slices_with_label_2 = \
                    (nifti.ndarray == 2).astype(np.int).sum(axis=0).sum(axis=0).astype(np.bool).sum()
                min_slices_with_label_1 = min(min_slices_with_label_1, slices_with_label_1)
                min_slices_with_label_2 = min(min_slices_with_label_2, slices_with_label_2)
                max_slices_with_label_1 = max(max_slices_with_label_1, slices_with_label_1)
                max_slices_with_label_2 = max(max_slices_with_label_2, slices_with_label_2)

                if slices_with_label_1 == 0:
                    files_without_label_1.append(int(result.groupdict()['id']))
                if slices_with_label_2 == 0:
                    files_without_label_2.append(int(result.groupdict()['id']))

                label_counter += 1

        if img_counter != label_counter:
            logger.warning(
                'Number of images %s does not match the number of labels %s', img_counter, label_counter)

        if verbose:
            table = [
                ['', 'value'],
                ['Files without label 1', files_without_label_1],
                ['Files without label 2', files_without_label_2],
                ['Total CT files', img_counter],
                ['Total segmentation files', label_counter]
            ]
            logger.info('\n%s', str(tabulate(table, headers="firstrow")))

            table = [
                ['', 'min', 'max'],
                ['Image value', min_image_val, max_image_val],
                ['Slices with label 1', min_slices_with_label_1, max_slices_with_label_1],
                ['Slices with label 2', min_slices_with_label_2, max_slices_with_label_2],
                ['Height', min_height, max_height],
                ['Width', min_width, max_width],
                ['Depth', min_depth, max_depth],
            ]
            logger.info('\n%s', str(tabulate(table, headers="firstrow")))

        return min_image_val, max_image_val, min_slices_with_label_1, max_slices_with_label_1, \
            min_slices_with_label_2, max_slices_with_label_2,  \
            files_without_label_1, files_without_label_2, min_height, min_width, min_depth, \
            max_height, max_width, max_depth, img_counter, label_counter

    def get_lowest_highest_bounds(self):
        """
        Analyzes all the volume NIFTI files from self.db_path and returns the lowest and highets bounds
        (from boxplots)

        Returns:
            lowest_bound<int>, highest_bound<int>
        """
        wildcard = os.path.join(self.db_path, 'volume-*.nii')
        files_ = glob.glob(wildcard)
        lowest_bound = float('inf')
        highest_bound = float('-inf')

        for file_ in tqdm(files_, unit='volume NIfTI files'):
            boxplot = BoxPlot(NIfTI(file_).ndarray)
            lower_bound, upper_bound = boxplot.find_quantiles_median_iqr_wiskers()[-2:]
            lowest_bound = min(lowest_bound, lower_bound)
            highest_bound = max(highest_bound, upper_bound)

        return lowest_bound, highest_bound

    def perform_visual_verification(
            self, subject_id: int, /, *, alpha: float = .4, scans: Union[int, list] = 1, clahe: bool = False,
            rgb_mask: tuple = None
    ):
        """
        Plots CTs along with its mask from the processed dataset located at self.saving_path

        To see the changes open visual_verification.png and it will be continuosly updated with new mask data

        Kwargs:
            subject_id  <int>: chosen id, without leading zeros, from the generated labels (label_<id>.nii.gz)
                               or CTs (CT_<id>.pro.nii.gz) to be analyzed.
            alpha     <float>: alpha channel value in range ]0, 1]. Default 0.9
            scans <int, list>: If set to an integer it will be the number of the scan to be analyzed. If
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

        label = NIfTI(os.path.join(self.saving_path, self.GEN_LABEL_FILENAME_TPL.format(subject_id)))
        ct = NIfTI(os.path.join(self.saving_path, self.GEN_CT_FILENAME_TPL.format(subject_id)))
        # print(ct.shape)
        scans = [scans] if isinstance(scans, int) else [*range(*scans)]

        for scan in scans:
            mask = label.ndarray[..., scan] * 255
            image = scale_using_general_min_max_values(
                ct.ndarray[..., scan].astype(float).clip(CT_MIN_VAL, CT_MAX_VAL),
                min_val=CT_MIN_VAL, max_val=CT_MAX_VAL, feats_range=(0, 255)
            )

            if clahe:
                image = np.asarray(Image.fromarray(np.uint8(image)).convert('L'))
                image = equalize_adapthist(image)*255
                image = image.astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image.astype(np.uint8)).convert('RGB')

            mask = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')
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
        Splits the processet LiTS17 into train, validation and test subdatasets

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

        labels_wildcard = os.path.join(self.saving_path, 'label_*.nii.gz')
        labels_files = glob.glob(labels_wildcard)
        labels_files.sort()
        cts_wildcard = os.path.join(self.saving_path, 'CT_*.nii.gz')
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
