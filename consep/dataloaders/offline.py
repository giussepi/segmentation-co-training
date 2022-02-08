# -*- coding: utf-8 -*-
""" consep/dataloaders/offline """

import os
import re

import numpy as np
import torch
from gtorch_utils.datasets.segmentation import DatasetTemplate
from PIL import Image


class OfflineCoNSePDataset(DatasetTemplate):
    """
    Dataset for the image CoNSeP dataset created through
    consep.processors.offline->CreateDataset

    Usage:

    train, val, test = OfflineCoNSePDataset.get_subdatasets(
        train_path='consep_dataset/train', val_path='consep_dataset/val')

    train_dataloader = DataLoader(
        train,
        num_workers=0,
        batch_size=batch_size * num_gpus,
        shuffle=run_mode == DB.TRAIN,
        drop_last=run_mode == DB.TRAIN,
        **SeedWorker(preserve_reproductibility=True)(),
    )
    """

    NUM_CLASSES = 1

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            images_masks_path  <str>: path to the folder for containing images and masks
            filename_reg       <str>: regular expression to get the index id from the crop filename.
                                      Default r'(?P<filename>\d+).ann.tiff'
            image_extension    <str>: image extension. Default '.ann.tiff'
            mask_extension     <str>: mask extension. Default '.mask.png'
            cot_mask_extension <str>: co-traning mask extension. Default '.cot.mask.png'
            cotraining        <bool>: If True the co-training masks are returned; otherwise, returns
                                      ground truth masks. Default False
            original_masks    <bool>: If original_masks == cotraining_mask == True, then both the original
                                      and cotraining masks are returned.
                                      Default False
        """
        self.images_masks_path = kwargs.get('images_masks_path')
        self.filename_reg = kwargs.get('filename_reg', r'(?P<filename>\d+).ann.tiff')
        self.image_extension = kwargs.get('image_extension', '.ann.tiff')
        self.mask_extension = kwargs.get('mask_extension', '.mask.png')
        self.cot_mask_extension = kwargs.get('cot_mask_extension', '.cot.mask.png')
        self.cotraining = kwargs.get('cotraining', False)
        self.original_masks = kwargs.get('original_masks', False)

        assert isinstance(self.images_masks_path, str), type(self.images_masks_path)
        assert os.path.isdir(self.images_masks_path), self.images_masks_path
        assert isinstance(self.filename_reg, str), type(self.filename_reg)
        assert isinstance(self.image_extension, str), type(self.image_extension)
        assert isinstance(self.mask_extension, str), type(self.mask_extension)
        assert isinstance(self.cot_mask_extension, str), type(self.cot_mask_extension)
        assert isinstance(self.cotraining, bool), type(self.cotraining)
        assert isinstance(self.original_masks, bool), type(self.original_masks)

        self.pattern = re.compile(self.filename_reg)
        self.image_list = [file_ for file_ in os.listdir(self.images_masks_path) if bool(self.pattern.fullmatch(file_))]

    def __len__(self):
        return len(self.image_list)

    def get_original_mask(self, idx):
        """
        Loads and returns the original mask of the image with index idx

        Kwargs:
            idx <int>: image index

        Returns:
            original_mask <np.ndarray>
        """
        assert isinstance(idx, int), type(idx)

        mask = Image.open(os.path.join(
            self.images_masks_path,
            self.image_list[idx].replace(self.image_extension, self.mask_extension)
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

        image = Image.open(os.path.join(self.images_masks_path, self.image_list[idx]))
        cot_mask_path = original_mask = original_target_mask = ''

        if self.cotraining:
            cot_mask_path = os.path.join(
                self.images_masks_path,
                self.image_list[idx].replace(self.image_extension, self.cot_mask_extension)
            )

            # the very first time the co-training files do not exits; thus,
            # we use the ground truth masks
            if os.path.isfile(cot_mask_path):
                mask = Image.open(cot_mask_path)
            else:
                mask = self.get_original_mask(idx)

            if self.original_masks:
                original_mask = self.get_original_mask(idx)
        else:
            mask = self.get_original_mask(idx)

        assert image.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {image.size} and {mask.size}'

        image = np.array(image.convert('RGB')) if image.mode != 'RGB' else np.array(image)
        mask = np.array(mask.convert('L')) if mask.mode != 'L' else np.array(mask)
        target_mask = np.zeros((*mask.shape[:2], self.NUM_CLASSES), dtype=np.float32)
        target_mask[..., 0] = (mask == 255).astype(np.float32)  # nuclei class

        if isinstance(original_mask, Image.Image):
            original_mask = np.array(original_mask.convert(
                'L')) if original_mask.mode != 'L' else np.array(original_mask)
            original_target_mask = np.zeros((*original_mask.shape[:2], self.NUM_CLASSES), dtype=np.float32)
            original_target_mask[..., 0] = (original_mask == 255).astype(np.float32)  # nuclei class

        return image, target_mask, '', '', cot_mask_path, original_target_mask

    @staticmethod
    def preprocess(img):
        """
        Preprocess the image and returns it

        Args:
            img <np.ndarray>:
        Returns:
            image <np.ndarray>
        """
        assert isinstance(img, np.ndarray), type(img)

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        if img.max() > 1:
            img = img / 255

        return img

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
        image = self.preprocess(image)
        mask = self.preprocess(mask)

        if isinstance(original_mask, np.ndarray):
            original_mask = self.preprocess(original_mask)
            original_mask = torch.from_numpy(original_mask).type(torch.FloatTensor)

        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'label': label,
            'label_name': label_name,
            'updated_mask_path': updated_mask_path,
            'original_mask': original_mask
        }

    @classmethod
    def get_subdatasets(cls, **kwargs):
        """
        Creates and returns train, validation and test subdatasets to be used with DataLoaders

        Kwargs:
            train_path         <str>: path to the folder for containing training images and masks
            val_path           <str>: path to the folder for containing validation images and masks
            test_path          <str>: path to the folder for containing testing images and masks. Default ''
            filename_reg       <str>: regular expression to get the index id from the crop filename.
                                      Default r'(?P<filename>\d+).ann.tiff'
            image_extension    <str>: image extension. Default '.ann.tiff'
            mask_extension     <str>: mask extension. Default '.mask.png'
            cot_mask_extension <str>: co-traning mask extension. Default '.cot.mask.png'
            cotraining        <bool>: If True the co-training masks are be returned; otherwise, returns
                                      ground truth masks. Default False
        Returns:
           train <OfflineCoNSePDataset>, validation <OfflineCoNSePDataset>, test <OfflineCoNSePDataset or None>
        """
        train_path = kwargs.pop('train_path')
        val_path = kwargs.pop('val_path')
        test_path = kwargs.pop('test_path', '')

        # making sure this keyword argument is not passed when creating the instances
        if 'images_masks_path' in kwargs:
            kwargs.pop('images_masks_path')

        train_dataset = cls(images_masks_path=train_path, **kwargs)
        val_dataset = cls(images_masks_path=val_path, **kwargs)
        test_dataset = cls(images_masks_path=test_path, **kwargs) if test_path else None

        return train_dataset, val_dataset, test_dataset
