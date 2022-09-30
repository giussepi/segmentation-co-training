# -*- coding: utf-8 -*-
""" lits17/processors/lits17cropmgr """

import glob
import operator
import os
import re
from functools import reduce
from itertools import chain
from random import sample
from typing import Tuple

import numpy as np
from gutils.decorators import timing
from gutils.folders import clean_create_folder
from gutils.images.images import NIfTI
from gutils.images.processing import get_slices_coords
from gutils.numpy_.numpy_ import scale_using_general_min_max_values
from logzero import logger
from tqdm import tqdm

from lits17.constants import CT_MIN_VAL, CT_MAX_VAL


__all__ = ['LiTS17CropMGR']


class LiTS17CropMGR:
    """
    Creates a new dataset of crops extracted from the outcome of LiTS17MGR

    Note: only tested with binary cases, but it should work for the multi-class lists17

    Usage:
        LiTS17CropMGR('<path_to_outcome_of_LiTS17MGR>')()
    """

    GEN_CT_FILENAME_TPL = 'CT_{:03d}_crop_{:03d}.nii.gz'
    GEN_MASK_FILENAME_TPL = 'label_{:03d}_crop_{:03d}.nii.gz'
    GENERIC_REGEX = r'.*(?P<type>\w+)_(?P<id>\d+).nii.gz'

    def __init__(
            self, db_path: str, /, *, patch_size: Tuple[int] = None, patch_overlapping: Tuple[float] = None,
            only_crops_with_masks: bool = True, min_mask_area: float = 25e-4, min_crop_mean: float = .41,
            crops_per_label: int = 20, saving_path: str = 'LiTS17-Pro-Crops', adjust_depth: bool = True,
            verbose: bool = True
    ):
        """
        Initializes the object instance

        Kwargs:
            db_path         <str>: Path to the main folder containing the dataset processed by LiTS17MGR
            patch_size    <tuple>: size of patches (height, width, scans). Default (80, 80, 32)
            patch_overlapping <tuple>: overlapping of patches (height, width, scans). Default (.25, .25, .25)
            only_crops_with_masks <bool>: If True, only crops containing non-zero labels are  extracted.
                                   Default True
            min_mask_area <float>: mininum percentage (in range ]0, 1[) of crop area containing the labels.
                                   i.e. The minimum area per label. Default 25e-4,
            min_crop_mean <float>: minimum mean of scaled image crops. This helps to avoid some crops
                                   containing mostly not relevant areas. Using the default configuration
                                   plus the flip transform we choose 0.41 as an acceptable default value.
                                   To disable this filter just set it to 0.
            crops_per_label <int>: maximum number of random crops per label. In some cases it could be less,
                                   so proper warning messages will be shown. Set it to -1 to use all crops.
                                   Default 20
            saving_path     <str>: Path to the folder where the processed labels and CTs will be stored
                                   Default LiTS17-Pro-Crops
            adjust_depth   <bool>: If set to False, the depth axis will not be modified when centering
                                   the crop mask. Default True
            verbose        <bool>: Whether or not print crops processing information. Default True
        """
        patch_size = patch_size if patch_size else (80, 80, 32)
        patch_overlapping = patch_overlapping if patch_overlapping else (.25, .25, .25)

        assert os.path.isdir(db_path), f'{db_path} does not point to a directory'
        assert isinstance(patch_size, tuple), type(patch_size)
        assert len(patch_size) == 3, 'patch_size must be tuple (height, width, scans)'
        assert isinstance(patch_overlapping, tuple), type(patch_overlapping)
        assert len(patch_overlapping) == 3, 'patch_overlapping must be tuple (height, width, scans)'
        assert isinstance(only_crops_with_masks, bool), type(only_crops_with_masks)
        assert 0 < min_mask_area < 1, min_mask_area
        assert 0 <= min_crop_mean < 1, min_crop_mean
        assert isinstance(crops_per_label, int), type(crops_per_label)
        assert crops_per_label == -1 or crops_per_label >= 1
        assert isinstance(saving_path, str), type(str)
        assert isinstance(adjust_depth, bool), type(adjust_depth)
        assert isinstance(verbose, bool), type(verbose)

        self.db_path = db_path
        self.patch_size = patch_size
        self.patch_overlapping = patch_overlapping
        self.only_crops_with_masks = only_crops_with_masks
        self.min_mask_area = min_mask_area
        self.min_crop_mean = min_crop_mean
        self.crops_per_label = crops_per_label
        self.saving_path = saving_path
        self.adjust_depth = adjust_depth
        self.verbose = verbose

        self.generic_pattern = re.compile(self.GENERIC_REGEX)
        self.overlapping = (np.array(self.patch_size) * np.array(self.patch_overlapping)).astype(int)
        # defining the min_area per scan/slide
        self.min_area = reduce(operator.mul, self.patch_size[:2]) * self.min_mask_area

    def __call__(self):
        """
        Generates the crops dataset and verifies all crops dimensions
        """
        self.process()
        self.verify_generated_db_dimensions()

    @timing
    def process(self):
        """
        Iterates all CTs and labels and created their corresponding crops
        """
        labels_wildcard = os.path.join(self.db_path, '**/label_*.nii.gz')
        labels_files = glob.glob(labels_wildcard, recursive=True)
        labels_files.sort()
        clean_create_folder(self.saving_path)
        total_crops = label2_crops = label1_crops = label0_crops = 0

        logger.info('Generating crops:')
        for label_file_path in tqdm(labels_files):
            if self.crops_per_label == -1:
                total_crops += self._create_all_crops(label_file_path)
            else:
                total, label2, label1, label0 = self._create_random_crops(label_file_path)
                total_crops += total
                label2_crops += label2
                label1_crops += label1
                label0_crops += label0

        if not self.verbose:
            return

        logger.info('Total crops created: %s', total_crops)

        if self.crops_per_label != -1:
            logger.info('Label 2 crops: %s', label2_crops)
            logger.info('Label 1 crops: %s', label1_crops)
            logger.info('Label 0 crops: %s', label0_crops)

    def _save_from_coords(
            self, img: np.ndarray, mask: np.ndarray, crops_dir: str,  label_id: int, crop_counter: int,
            iy: int, ix: int, iz: int):
        """
        Saves the CT and segmenation crops based on the provided coordinates

        Kwargs:
            img   <np.ndarray>:
            mask  <np.ndarray>:
            crops_dir    <str>:
            label_id     <int>:
            crop_counter <int>:
            iy           <int>: y crop coordinate
            ix           <int>: x crop coordinate
            iz           <int>: z crop coordinate
        """
        img_crop = img.ndarray[
            iy: iy+self.patch_size[0], ix:ix+self.patch_size[1], iz:iz+self.patch_size[2]]
        mask_crop = mask.ndarray[
            iy: iy+self.patch_size[0], ix:ix+self.patch_size[1], iz:iz+self.patch_size[2]]
        NIfTI.save_numpy_as_nifti(
            img_crop,
            os.path.join(crops_dir, self.GEN_CT_FILENAME_TPL.format(label_id, crop_counter))
        )
        NIfTI.save_numpy_as_nifti(
            mask_crop,
            os.path.join(crops_dir, self.GEN_MASK_FILENAME_TPL.format(label_id, crop_counter))
        )

    def _create_all_crops(self, label_file_path: str):
        """
        Creates crops using the sliding window technique

        Kwargs:
            label_file_path <str>: path to the label file

        Returns:
            total_crops<int>
        """
        assert os.path.isfile(label_file_path)

        result = self.generic_pattern.match(label_file_path)

        if result is None:
            raise RuntimeError(f'The GENERIC_REGED did not match {label_file_path}')

        label_id = int(result.groupdict()['id'])
        img = NIfTI(label_file_path.replace('label', 'CT'))
        mask = NIfTI(label_file_path)

        assert img.shape == mask.shape, 'img.shape is not equal to mask.shape'

        crops_dir = os.path.join(
            os.path.dirname(label_file_path).replace(self.db_path, self.saving_path),
            f'{label_id:03d}'
        )
        clean_create_folder(crops_dir)
        mask_crop = img_crop = None
        crop_counter = 0

        for iy in get_slices_coords(img.shape[0], self.patch_size[0], self.overlapping[0]):
            for ix in get_slices_coords(img.shape[1], self.patch_size[1], self.overlapping[1]):
                for iz in get_slices_coords(img.shape[2], self.patch_size[2], self.overlapping[2]):
                    mask_crop = mask.ndarray[
                        iy: iy+self.patch_size[0], ix:ix+self.patch_size[1], iz:iz+self.patch_size[2]]
                    img_crop = img.ndarray[
                        iy: iy+self.patch_size[0], ix:ix+self.patch_size[1], iz:iz+self.patch_size[2]]
                    scaled_img_crop = scale_using_general_min_max_values(
                        img_crop.clip(CT_MIN_VAL, CT_MAX_VAL), min_val=CT_MIN_VAL, max_val=CT_MAX_VAL,
                        feats_range=(0, 1)
                    )

                    # making sure the scaled img crop mean is bigger or equal to min_crop_mean
                    if scaled_img_crop.mean() >= self.min_crop_mean:
                        # making sure that at least one mask slice ([height, width]) contains
                        # one crop with a label area bigger or equal than min_area
                        if (mask_crop == 2).sum(axis=-1).astype(bool).sum() >= self.min_area:
                            ciy, cix, ciz = self.get_centred_mask_crop(mask_crop, 2, iy, ix, iz, mask.shape)
                            crop_counter += 1
                            self._save_from_coords(img, mask, crops_dir, label_id, crop_counter, ciy, cix, ciz)
                        if (mask_crop == 1).sum(axis=-1).astype(bool).sum() >= self.min_area:
                            ciy, cix, ciz = self.get_centred_mask_crop(mask_crop, 1, iy, ix, iz, mask.shape)
                            crop_counter += 1
                            self._save_from_coords(img, mask, crops_dir, label_id, crop_counter, ciy, cix, ciz)
                        if self.only_crops_with_masks:
                            continue
                        if (mask_crop == 0).sum(axis=-1).astype(bool).sum() >= self.min_area:
                            crop_counter += 1
                            self._save_from_coords(img, mask, crops_dir, label_id, crop_counter, iy, ix, iz)

        return crop_counter

    def get_centred_mask_crop(
            self, mask_crop: np.ndarray, label: int, iy: int, ix: int, iz: int, mask_shape: tuple):
        """
        Centers the mask and returns the new crop coordinates

        Kwargs:
            mask_crop <np.ndarray>:
            label        <int>:
            iy           <int>: y crop coordinate
            ix           <int>: x crop coordinate
            iz           <int>: z crop coordinate
            mask_shape <tuple>: shape of the original mask utilised to create the mask_crop
        Returns:
            new_iy<int>, new_ix<int>, new_iz<int>
        """
        assert isinstance(mask_crop, np.ndarray), type(mask_crop)
        assert isinstance(label, int), type(label)
        assert isinstance(iy, int), type(iy)
        assert isinstance(ix, int), type(ix)
        assert isinstance(iz, int), type(iz)
        assert isinstance(mask_shape, tuple), type(mask_shape)
        assert len(mask_shape) == 3, len(mask_shape)

        merged_height_width_mask = (mask_crop == label).sum(axis=-1)
        merged_depth_mask = (mask_crop == label).sum(axis=0).sum(axis=0)
        labelled_y = merged_height_width_mask.sum(axis=0).nonzero()[0]
        relative_min_y, relative_max_y = labelled_y[0], labelled_y[-1]
        labelled_x = merged_height_width_mask.sum(axis=1).nonzero()[0]
        relative_min_x, relative_max_x = labelled_x[0], labelled_x[-1]
        labelled_z = merged_depth_mask.nonzero()[0]
        relative_min_z, relative_max_z = labelled_z[0], labelled_z[-1]

        mask_bbox_height = relative_max_y - relative_min_y
        mask_bbox_width = relative_max_x - relative_min_x
        mask_bbox_depth = relative_max_z - relative_min_z

        absolute_min_y = iy + relative_min_y
        absolute_min_x = ix + relative_min_x
        absolute_min_z = iz + relative_min_z

        # problem located hereee
        new_iy = max(0, absolute_min_y - (self.patch_size[0] - mask_bbox_height) // 2)

        if new_iy + self.patch_size[0] > mask_shape[0]:
            new_iy = mask_shape[0] - self.patch_size[0]

        new_ix = max(0, absolute_min_x - (self.patch_size[1] - mask_bbox_width) // 2)

        if new_ix + self.patch_size[1] > mask_shape[1]:
            new_ix = mask_shape[1] - self.patch_size[1]

        if self.adjust_depth:
            new_iz = max(0, absolute_min_z - (self.patch_size[2] - mask_bbox_depth) // 2)

            if new_iz + self.patch_size[2] > mask_shape[2]:
                new_iz = mask_shape[2] - self.patch_size[2]
        else:
            new_iz = iz

        return new_iy, new_ix, new_iz

    def _create_random_crops(self, label_file_path: str):
        """
        Randomly selects up to a  user-defined number of crops (could be less) from a pool of
        crops coordinates per label created using the sliding window technique

        Kwargs:
            label_file_path <str>: path to the label file

        Returns:
            total_crops<int>, label2_crops<int>, label1_crops<int>, label0_crops<int>
        """
        assert os.path.isfile(label_file_path)

        result = self.generic_pattern.match(label_file_path)

        if result is None:
            raise RuntimeError(f'The GENERIC_REGED did not match {label_file_path}')

        label_id = int(result.groupdict()['id'])
        img = NIfTI(label_file_path.replace('label', 'CT'))
        mask = NIfTI(label_file_path)

        assert img.shape == mask.shape, 'img.shape is not equal to mask.shape'

        crops_dir = os.path.join(
            os.path.dirname(label_file_path).replace(self.db_path, self.saving_path),
            f'{label_id:03d}'
        )
        clean_create_folder(crops_dir)
        mask_crop = img_crop = None
        crop_counter = 0

        two_labels = []
        one_labels = []
        zero_labels = []

        for iy in get_slices_coords(img.shape[0], self.patch_size[0], self.overlapping[0]):
            for ix in get_slices_coords(img.shape[1], self.patch_size[1], self.overlapping[1]):
                for iz in get_slices_coords(img.shape[2], self.patch_size[2], self.overlapping[2]):
                    mask_crop = mask.ndarray[
                        iy: iy+self.patch_size[0], ix:ix+self.patch_size[1], iz:iz+self.patch_size[2]]
                    img_crop = img.ndarray[
                        iy: iy+self.patch_size[0], ix:ix+self.patch_size[1], iz:iz+self.patch_size[2]]
                    scaled_img_crop = scale_using_general_min_max_values(
                        img_crop.clip(CT_MIN_VAL, CT_MAX_VAL), min_val=CT_MIN_VAL, max_val=CT_MAX_VAL,
                        feats_range=(0, 1)
                    )

                    # making sure the scaled img crop mean is bigger or equal to min_crop_mean
                    if scaled_img_crop.mean() >= self.min_crop_mean:
                        # making sure that at least one mask slice ([height, width]) contains
                        # one crop with a label area bigger or equal than min_area
                        if (mask_crop == 2).sum(axis=-1).astype(bool).sum() >= self.min_area:
                            two_labels.append(self.get_centred_mask_crop(mask_crop, 2, iy, ix, iz, mask.shape))
                        if (mask_crop == 1).sum(axis=-1).astype(bool).sum() >= self.min_area:
                            one_labels.append(self.get_centred_mask_crop(mask_crop, 1, iy, ix, iz, mask.shape))
                        if self.only_crops_with_masks:
                            continue
                        if (mask_crop == 0).sum(axis=-1).astype(bool).sum() >= self.min_area:
                            zero_labels.append((iy, ix, iz))

        # not all the time we process datasets with liver and lesion lesions
        if two_labels:
            try:
                two_labels = sample(two_labels, k=self.crops_per_label)
            except ValueError as e:
                logger.warning('%s - label 2 - total label files %s: %s', label_file_path, len(two_labels), e)

        try:
            one_labels = sample(one_labels, k=self.crops_per_label)
        except ValueError as e:
            logger.warning('%s - label 1 - total label files %s: %s', label_file_path, len(one_labels), e)

        if zero_labels:
            try:
                zero_labels = sample(zero_labels, k=self.crops_per_label)
            except ValueError as e:
                logger.warning('%s - label 0 - total label files %s: %s', label_file_path, len(zero_labels), e)

        for iy, ix, iz in chain(two_labels, one_labels, zero_labels):
            crop_counter += 1
            self._save_from_coords(img, mask, crops_dir, label_id, crop_counter, iy, ix, iz)

        return crop_counter, len(two_labels), len(one_labels), len(zero_labels)

    def verify_generated_db_dimensions(self):
        """
        Verifies all the generated crops have the right patch size
        """
        wildcard = os.path.join(self.saving_path, '**/*.nii.gz')
        file_list = glob.glob(wildcard, recursive=True)

        logger.info('Verifying dimensions from generated crops')
        for file_path in tqdm(file_list, unit='NIfTI files'):
            nifti = NIfTI(file_path)
            assert nifti.shape == self.patch_size, (file_path, nifti.shape)
