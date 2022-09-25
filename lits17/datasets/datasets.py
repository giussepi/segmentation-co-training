# -*- coding: utf-8 -*-
""" lits17/datasets/datasets """

import glob
import os
import re

import numpy as np
from gutils.images.images import NIfTI

from lits17.datasets.base import BaseLiTS17Dataset


__all__ = ['LiTS17Dataset', 'LiTS17CropDataset']


class LiTS17Dataset(BaseLiTS17Dataset):
    """
    Dataset for the LiTS17 dataset created through
    lits17.processors.LiTS17MGR

    Usage:
        train, val, test = LiTS17Dataset.get_subdatasets(
            train_path='LiTS17-Pro/train', val_path='LiTS17-Pro/val', test_path='LiTS17-Pro/test')
        train_dataloader = DataLoader(
            train,
            num_workers=0,
            batch_size=batch_size * num_gpus,
            shuffle=run_mode == DB.TRAIN,
            drop_last=run_mode == DB.TRAIN,
        )
    """


class LiTS17CropDataset(BaseLiTS17Dataset):
    """
    Dataset for the LiTS17 dataset created through
    lits17.processors.LiTS17CropMGR

    Usage:
        train, val, test = LiTS17CropDataset.get_subdatasets(
            train_path='LiTS17-Pro-Crops/train',
            val_path='LiTS17-Pro-Crops/val',
            test_path='LiTS17-Pro-Crops/test'
        )
        train_dataloader = DataLoader(
            train,
            num_workers=0,
            batch_size=batch_size * num_gpus,
            shuffle=run_mode == DB.TRAIN,
            drop_last=run_mode == DB.TRAIN,
        )
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            images_masks_path  <str>: path to the folder for containing images and masks
            filename_reg       <str>: regular expression to get the index id from the crop filename.
                                      Default r'^.*CT_(?P<id>\d+)_crop_(?P<crop>\d+).nii.gz$'
            mask_name_tpl      <str>: Template to build the label name using the id.
                                      Default 'label_{:03d}_crop_{:03d}.nii.gz'
            cot_mask_name_tpl  <str>: Template to build the co-training label name using the id.
                                      Default 'label_{:03d}_crop_{:03d}.cot.nii.gz'
            cotraining        <bool>: If True the co-training masks are returned; otherwise, returns
                                      ground truth masks. Default False
            original_masks    <bool>: If original_masks == cotraining_mask == True, then both the original
                                      and cotraining masks are returned.
                                      Default False
            transform <callable, None>: Data augmentation transforms. See ct82.settings.
                                      Defaullt None
            cache             <bool>: If True all the images will be cached. Default False
        """
        self.images_masks_path = kwargs.get('images_masks_path')
        self.filename_reg = kwargs.get('filename_reg', r'^.*CT_(?P<id>\d+)_crop_(?P<crop>\d+).nii.gz$')
        self.mask_name_tpl = kwargs.get('mask_name_tpl', 'label_{:03d}_crop_{:03d}.nii.gz')
        self.cot_mask_name_tpl = kwargs.get('cot_mask_name_tpl', 'label_{:03d}_crop_{:03d}.cot.nii.gz')
        self.cotraining = kwargs.get('cotraining', False)
        self.original_masks = kwargs.get('original_masks', False)
        self.transform = kwargs.get('transform', None)
        self.cache = kwargs.get('cache', False)

        assert isinstance(self.images_masks_path, str), type(self.images_masks_path)
        assert os.path.isdir(self.images_masks_path), self.images_masks_path
        assert isinstance(self.filename_reg, str), type(self.filename_reg)
        assert isinstance(self.mask_name_tpl, str), type(self.mask_name_tpl)
        assert isinstance(self.cot_mask_name_tpl, str), type(self.cot_mask_name_tpl)
        assert isinstance(self.cotraining, bool), type(self.cotraining)
        assert isinstance(self.original_masks, bool), type(self.original_masks)
        assert isinstance(self.cache, bool), type(self.cache)

        if self.transform is not None:
            assert callable(self.transform)

        self.pattern = re.compile(self.filename_reg)
        self.image_list = glob.glob(os.path.join(self.images_masks_path, '**/CT_*.nii.gz'), recursive=True)

        self.cached = {}

    def get_original_mask(self, idx):
        """
        Loads and returns the original mask of the image with index idx

        Kwargs:
            idx <int>: image index

        Returns:
            original_mask <NIfTI>
        """
        assert isinstance(idx, int), type(idx)

        result = self.pattern.fullmatch(self.image_list[idx]).groupdict()
        subject_id, crop_id = int(result['id']), int(result['crop'])
        mask = NIfTI(os.path.join(
            os.path.dirname(self.image_list[idx]),
            self.mask_name_tpl.format(subject_id, crop_id)
        ))

        return mask

    def get_image_and_mask_files(self, idx):
        """
        Loads the image and mask corresponding to the file at position idx in the image list.
        Besides, both are properly formatted to be used by the neuronal network before
        returning them.

        Kwargs:
            idx <int>: image index

        Returns:
            image <np.ndarray>, target_mask <np.ndarray>, '', '', co_training_mask_path <str>,
            original_target_mask <np.ndarray or empty string>
        """
        assert isinstance(idx, int), type(idx)

        if self.cache and idx in self.cached:
            return self.cached[idx]

        image = NIfTI(self.image_list[idx])
        cot_mask_path = original_mask = original_target_mask = ''

        if self.cotraining:
            result = self.pattern.fullmatch(self.image_list[idx])
            subject_id, crop_id = int(result.groupdict()['id']), int(result.groupdict()['crop'])
            cot_mask_path = os.path.join(
                os.path.dirname(self.image_list[idx]),
                self.cot_mask_name_tpl.format(subject_id, crop_id)
            )

            # the very first time the co-training files do not exits; thus,
            # we use the ground truth masks
            if os.path.isfile(cot_mask_path):
                mask = NIfTI(cot_mask_path)
            else:
                mask = self.get_original_mask(idx)

            if self.original_masks:
                original_mask = self.get_original_mask(idx)
        else:
            mask = self.get_original_mask(idx)

        assert image.shape == mask.shape, \
            (f'Image and mask {idx} should be the same shape, but they are {image.shape} and ',
             f'{mask.shape} respectively')

        image = image.ndarray.copy()
        mask = mask.ndarray.copy()
        # TODO: review if this is correct
        target_mask = mask.astype(np.float32)

        if isinstance(original_mask, NIfTI):
            original_mask = original_mask.ndarray.copy()

        if self.cache:
            self.cached[idx] = (image, target_mask, '', '', cot_mask_path, original_target_mask)

        return image, target_mask, '', '', cot_mask_path, original_target_mask
