# -*- coding: utf-8 -*-
""" lits17/datasets/base """

import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
from gtorch_utils.datasets.segmentation import DatasetTemplate
from gutils.images.images import NIfTI
from gutils.numpy_.numpy_ import scale_using_general_min_max_values
from logzero import logger

from lits17.constants import CT_MAX_VAL, CT_MIN_VAL
from lits17.settings import TRANSFORMS


__all__ = ['BaseLiTS17Dataset']


class BaseLiTS17Dataset(DatasetTemplate):
    """
    Base dataset class to create LiTS17 dataset classes to be used with datasets created through
    lits17.processors.LiTS17MGR

    Usage:
        class MyLiTS17Dataset(BaseLiTS17Dataset):
            ...

        train, val, test = MyLiTS17Dataset.get_subdatasets(
            train_path='LiTS17-Pro/train', val_path='LiTS17-Pro/val', test_path='LiTS17-Pro/test')
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
                                      Default r'^.*CT\_(?P<id>[\d]+).nii.gz$'
            mask_name_tpl      <str>: Template to build the label name using the id. Default 'label_{}.nii.gz'
            cot_mask_name_tpl  <str>: Template to build the co-training label name using the id.
                                      Default 'label_{}.cot.nii.gz'
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
        self.filename_reg = kwargs.get('filename_reg', r'^.*CT\_(?P<id>[\d]+).nii.gz$')
        self.mask_name_tpl = kwargs.get('mask_name_tpl', 'label_{}.nii.gz')
        self.cot_mask_name_tpl = kwargs.get('cot_mask_name_tpl', 'label_{}.cot.nii.gz')
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

    def __len__(self):
        return len(self.image_list)

    def get_original_mask(self, idx):
        """
        Loads and returns the original mask of the image with index idx

        Kwargs:
            idx <int>: image index

        Returns:
            original_mask <NIfTI>
        """
        assert isinstance(idx, int), type(idx)

        subject_id = self.pattern.fullmatch(self.image_list[idx]).groupdict()['id']
        mask = NIfTI(os.path.join(
            os.path.dirname(self.image_list[idx]),
            self.mask_name_tpl.format(subject_id)
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
            subject_id = self.pattern.fullmatch(self.image_list[idx]).groupdict()['id']
            cot_mask_path = os.path.join(
                os.path.dirname(self.image_list[idx]),
                self.cot_mask_name_tpl.format(subject_id)
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

    @staticmethod
    def preprocess(img: np.ndarray = None, mask: np.ndarray = None):
        """
        Preprocess the image and mask and returns them

        Kwargs:
            img  <np.ndarray, None>: 3D CT Scan in numpy format [height, width, channels]
            mask <np.ndarray, None>: 3D label in numpy format [height, width, channels]

        Returns:
            image <np.ndarray, None>, mask <np.ndarray, None>
        """
        if img is not None:
            assert isinstance(img, np.ndarray), type(img)
            img = img.transpose(2, 0, 1)[np.newaxis, ...]
            # TODO: maybe I shouldn't be using DICOM min and max values....
            img = scale_using_general_min_max_values(
                img.clip(CT_MIN_VAL, CT_MAX_VAL), min_val=CT_MIN_VAL, max_val=CT_MAX_VAL, feats_range=(0, 1))
        if mask is not None:
            assert isinstance(mask, np.ndarray), type(mask)
            mask = mask.transpose(2, 0, 1)[np.newaxis, ...]

        return img, mask

    def __getitem__(self, idx):
        """
        Gets the image and mask for the index idx and returns them

        Kwargs:
            idx <int>: image index

        Returns:
            dict(
                image=<Tensor>, mask=<Tensor>, label=<int> label_name=<str>, updated_mask_path <str>,
                original_mask <Tensor or empty str>
            )
        """
        assert isinstance(idx, int), type(idx)

        image, mask, label, label_name, updated_mask_path, original_mask = self.get_image_and_mask_files(idx)
        image, mask = self.preprocess(image, mask)

        # FIXME: I'm not sure applying online data augmentation will be compatible with
        #        cotraining. Maybe it will if for validation I only apply ToTensor and TypeCast
        #        double-check when and how co-training is applied
        if self.transform:
            # NOTE: for this dataset we are not using labels or label_names, but generally,
            #       creating crops that could not contain the mask would be a problem
            #       that would require extra work to determine wich label correspond to
            #       each crop before returning them. One workaround for binary problems
            #       is making sure that all the crops contains part or the mask.
            #       for multi-class or multilabel more processing will be required
            #       to determine the right label(s) per crop
            results = self.transform(dict(img=image, mask=mask))

            if isinstance(results, dict):
                # when using RandSpatialCropd
                image, mask = results['img'], results['mask']
            else:
                # when using RandCropByPosNegLabeld
                img_list, masks_list = [], []

                for result in results:
                    img_list.append(result['img'])
                    masks_list.append(result['mask'])

                image, mask = torch.stack(img_list), torch.stack(masks_list)

        if isinstance(original_mask, np.ndarray):
            # TODO: I'm not applying the transforms to the original_mask so
            # this will be a issue when applying co-training
            original_mask = self.preprocess(mask=original_mask)
            original_mask = torch.from_numpy(original_mask).type(torch.FloatTensor)

        return {
            'image': image,
            'mask': mask,
            'label': label,
            'label_name': label_name,
            'updated_mask_path': updated_mask_path,
            'original_mask': original_mask
        }

    @classmethod
    def get_subdatasets(
            cls, train_path: str = 'CT-82-Pro/train', val_path: str = 'CT-82-Pro/val',
            test_path: str = 'CT-82-Pro/test', ** kwargs
    ):
        """
        Creates and returns train, validation and test subdatasets to be used with DataLoaders

        Kwargs:
            train_path         <str>: path to the folder for containing training images and masks.
                                      Default 'CT-82-Pro/train'
            val_path           <str>: path to the folder for containing validation images and masks
                                      Default 'CT-82-Pro/val'
            test_path          <str>: path to the folder for containing testing images and masks.
                                      Default 'CT-82-Pro/test'
            filename_reg       <str>: regular expression to get the index id from the crop filename.
                                      Default r'^CT\_(?P<id>[\d]+).pro.nii.gz$'
            mask_name_tpl      <str>: Template to build the label name using the id. Default 'label_{}.nii.gz'
            cot_mask_name_tpl  <str>: Template to build the co-training label name using the id.
                                      Default 'label_{}.cot.nii.gz'
            cotraining        <bool>: If True the co-training masks are be returned; otherwise, returns
                                      ground truth masks. Default False,
            original_masks    <bool>: If original_masks == cotraining_mask == True, then both the original
                                      and cotraining masks are returned.
                                      Default False
        Returns:
           train <CT82Dataset>, validation <CT82Dataset>, test <CT82Dataset, None>
        """
        # making sure this keyword argument is not passed when creating the instances
        if 'images_masks_path' in kwargs:
            kwargs.pop('images_masks_path')
            logger.warning('The key images_masks_path has been removed from the kwargs')

        if 'transform' in kwargs:
            kwargs.pop('transform')
            logger.warning('The key transform has been removed from the kwargs')

        if TRANSFORMS is not None:
            assert isinstance(TRANSFORMS, dict), type(TRANSFORMS)
            assert 'train' in TRANSFORMS, 'TRANSFORMS does not have the key "train"'
            assert 'valtest' in TRANSFORMS, 'TRANSFORMS does not have the key "valtest"'
            transforms = TRANSFORMS
        else:
            transforms = defaultdict(lambda: None)

        train_dataset = cls(images_masks_path=train_path, transform=transforms['train'], **kwargs)
        val_dataset = cls(images_masks_path=val_path, transform=transforms['valtest'], **kwargs)
        test_dataset = cls(images_masks_path=test_path,
                           transform=transforms['valtest'], **kwargs) if test_path else None

        return train_dataset, val_dataset, test_dataset
