# -*- coding: utf-8 -*-
""" consep/dataloaders/train_loader """

import random

import numpy as np
import torch
import torch.utils.data
from gtorch_utils.constants import DB
from imgaug import augmenters as iaa

from consep.utils.utils import cropping_center
from consep.utils.augmentations import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


class FileLoader(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list and
    performs augmentation with the albumentation library.

    Args:
        file_list       <list>: list of filenames to load
        input_shape    <tuple>: shape of the network input [h,w] - defined in config.py
        mask_shape     <tuple>: shape of the network output [h,w] - defined in config.py
        mode             <str>: subdataset (see gtorch_utils.constants.py -> DB class). Default DB.TRAIN
        setup_augmentor <bool>: Whether or not perform data augmentation. Default=True

    Inspired on: https://github.com/vqdang/hover_net/blob/master/dataloader/train_loader.py
    """

    def __init__(self, **kwargs):
        """ Initializes the object instance """

        self.info_list = kwargs.get('file_list')
        self.input_shape = kwargs.get('input_shape')
        self.mask_shape = kwargs.get('mask_shape')
        self.mode = kwargs.get('mode', DB.TRAIN)
        setup_augmentor = kwargs.get('setup_augmentor', True)

        assert isinstance(self.info_list, list), type(self.info_list)
        assert len(self.info_list) > 0, 'info_list cannot be empty'
        assert isinstance(self.input_shape, tuple), type(self.input_shape)
        assert isinstance(self.mask_shape, tuple), type(self.mask_shape)
        DB.clean_subdataset_name(self.mode)

        self.id = 0
        self.shape_augs = self.input_augs = None

        if setup_augmentor:
            self.setup_augmentor(0, 0)

    def __len__(self):
        return len(self.info_list)

    def __get_augmentation(self, mode, rng):
        if mode == DB.TRAIN:
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode in (DB.VALIDATION, DB.TEST):
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs

    def setup_augmentor(self, worker_id, seed):
        """ Configurates the data augmentation procedure """
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id

    def __getitem__(self, idx):
        """
        Loads the image and annotations/mask from the object with id=idx, performs data augmentation,
        extracts crops from the centre, and returns a dictionary with the image and its binary mask

        Returns:
            {'img': img, 'mask': binary_mask}
        """
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}
        inst_map = (ann[..., 0]).copy()  # HW1 -> HW
        inst_map[inst_map > 0] = 1
        feed_dict["mask"] = cropping_center(inst_map, self.mask_shape)

        return feed_dict


class SeedWorker:
    """
    Returns arguments for the DataLoader to produce deterministic or stochastic results

    Usage:
        DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            **SeedWorker(preserve_reproductibility=True)()
        )
    """

    def __init__(self, preserve_reproductibility=False, seed=10):
        """
        Initializes the object instance

        Args:
            preserve_reproductibility <bool>: Whether or not have reproductible results. Default False
            seed <int>: Seed for the generator created when preserve_reproductibility = True
        """
        assert isinstance(preserve_reproductibility, bool), type(preserve_reproductibility)
        assert isinstance(seed, int), type(seed)
        assert seed >= 0

        self.preserve_reproductibility = preserve_reproductibility
        self.seed = seed

    def __call__(self):
        """ Functor call """
        if self.preserve_reproductibility:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

            return {
                'worker_init_fn': self.deterministic,
                'generator': generator
            }

        return {'worker_init_fn': self.nondeterministic}

    @staticmethod
    def deterministic(worked_id):
        """
        Initializes the seed to preserve reproductibility

        Source: https://pytorch.org/docs/master/notes/randomness.html#dataloader
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def nondeterministic(worker_id):
        """
        Makes sure the initialization are truly random

        Initializes the augmentor per worker, else duplicated rng generators may happen

        Source: https://github.com/vqdang/hover_net/blob/master/run_train.py#L48
        """
        # ! to make the seed chain reproducible, must use the torch random, not numpy
        # the torch rng from main thread will regenerate a base seed, which is then
        # copied into the dataloader each time it created (i.e start of each epoch)
        # then dataloader with this seed will spawn worker, now we reseed the worker
        worker_info = torch.utils.data.get_worker_info()
        # to make it more random, simply switch torch.randint to np.randint
        worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
        # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
        # retrieve the dataset copied into this worker process
        # then set the random seed for each augmentation
        #
        worker_info.dataset.setup_augmentor(worker_id, worker_seed)
