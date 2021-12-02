# -*- coding: utf-8 -*-
""" consep/processors/offline """

import os
import glob
from collections import namedtuple

from gtorch_utils.constants import DB
from gutils.decorators import timing
from gutils.folders import clean_create_folder
from logzero import logger
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from consep.dataloaders import OnlineCoNSePDataset, SeedWorker


DBObj = namedtuple('DBObj', ['mode', 'path'])


class CreateDataset:
    """
    Processes the patches from CoNSeP dataset created by consep.utils.patches.patches.ProcessDataset
    and creates image and mask crops

    Usage:
        CreateDataset(
            train_path='dataset/training_data/consep/train/540x540_164x164',
            val_path='dataset/training_data/consep/valid/540x540_164x164',
        )()
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            train_path        <str>: path to the folder containing the training NumPy files created by
                                     consep.utils.patches.patches.ProcessDataset
            val_path          <str>: path to the folder containing the validation NumPy files created by
                                     consep.utils.patches.patches.ProcessDataset
            test_path         <str>: path to the folder containing the testing NumPy files created by
                                     consep.utils.patches.patches.ProcessDataset. Default ''
            saving_path       <str>: path to the folder to save the images and masks. Default 'consep_dataset'
            num_gpus          <int>: number of GPUs available. Default 0
            num_workers       <int>: number of workers to be used with the Dataloader. Default 0
            crop_img_shape  <tuple>: Shape [h,w] to crop from the centre of the image. Default (270, 270)
            crop_mask_shape <tuple>: Shape [h,w] to crop from the centre of the mask. Default (270, 270)
            img_format        <str>: Saving image extension. Default 'tiff'
            mask_format       <str>: Saving mask extension. Default 'png'
            batch_size        <int>: batch size. Default 16
            setup_augmentor  <bool>: Whether or not perform data augmentation. Default=True
        """
        self.train_path = kwargs.get('train_path')
        self.val_path = kwargs.get('val_path')
        self.test_path = kwargs.get('test_path', '')
        self.saving_path = kwargs.get('saving_path', 'consep_dataset')
        self.num_gpus = kwargs.get('num_gpus')
        self.num_workers = kwargs.get('num_workers', 0)
        self.crop_img_shape = kwargs.get('crop_img_shape', (270, 270))
        self.crop_mask_shape = kwargs.get('crop_mask_shape', (270, 270))
        self.img_format = kwargs.get('img_format', 'tiff')
        self.mask_format = kwargs.get('mask_format', 'png')
        self.batch_size = kwargs.get('batch_size', 16)
        self.setup_augmentor = kwargs.get('setup_augmentor', True)

        assert os.path.isdir(self.train_path)
        assert os.path.isdir(self.val_path)
        if self.test_path:
            assert os.path.isdir(self.test_path)
        assert isinstance(self.saving_path, str), type(self.saving_path)
        assert self.saving_path != '', 'saving_path cannot be an empty string'
        assert isinstance(self.num_gpus, int), type(self.num_gpus)
        assert isinstance(self.num_workers, int), type(self.num_workers)
        assert self.num_workers >= 0, 'num_workers must be a positive integer'
        assert isinstance(self.crop_img_shape, tuple), type(self.crop_img_shape)
        assert isinstance(self.crop_mask_shape, tuple), type(self.crop_mask_shape)
        assert isinstance(self.img_format, str), type(self.img_format)
        assert isinstance(self.mask_format, str), type(self.mask_format)
        assert isinstance(self.batch_size, int), type(self.batch_size)
        assert self.batch_size > 0
        assert isinstance(self.setup_augmentor, bool), type(self.setup_augmentor)

        self.num_gpus = self.num_gpus if self.num_gpus > 0 else 1

    def __call__(self):
        self.process_dataset()

    def process_subdataset(self, db_obj):
        """
        Processes a consep subdataset creating its crop iamges and masks

        Args:
            db_obj <DBObj>: DBObj instance containing the mode and path
        """
        assert isinstance(db_obj, DBObj), type(db_obj)

        # TODO: maybe I should pass the paths for train val and test...
        train_list = glob.glob(os.path.join(db_obj.path, '*.npy'))
        train_list.sort()
        total_batch_size = self.batch_size * self.num_gpus

        dataset = OnlineCoNSePDataset(
            file_list=train_list,
            input_shape=self.crop_img_shape,
            mask_shape=self.crop_mask_shape,
            mode=db_obj.mode,
            setup_augmentor=self.setup_augmentor,
        )

        dataloader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=total_batch_size,
            shuffle=False,
            drop_last=False,
            **SeedWorker(preserve_reproductibility=True)(),
        )

        clean_create_folder(os.path.join(self.saving_path, db_obj.mode))
        counter = 0

        for batch in tqdm(dataloader, desc='Creating crops', unit='batch'):
            for i in range(total_batch_size):
                counter += 1

                # dealing with the case when the last returned batch has less elements
                # than total_batch_size
                try:
                    img = Image.fromarray(batch['img'][i, :].detach().cpu().numpy())
                    mask = batch['mask'][i, :].detach().cpu().numpy()
                except IndexError:
                    break
                else:
                    mask[mask == 1] = 255
                    mask = Image.fromarray(mask).convert('L')
                    img.save(os.path.join(
                        self.saving_path, db_obj.mode, f'{counter}.ann.{self.img_format}'))
                    mask.save(os.path.join(
                        self.saving_path, db_obj.mode, f'{counter}.mask.{self.mask_format}'))

    @timing
    def process_dataset(self):
        """ Processes all the consep subdataset creating their crop images and masks """
        subdatasets = [DBObj(DB.TRAIN, self.train_path), DBObj(DB.VALIDATION, self.val_path)]

        if self.test_path:
            subdatasets.append(DBObj(DB.TEST, self.test_path))

        for subdataset in subdatasets:
            logger.info(f'Processing {subdataset.mode} subdataset.')
            self.process_subdataset(subdataset)
