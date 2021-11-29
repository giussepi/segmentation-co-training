# -*- coding: utf-8 -*-
""" main """

import glob
import logzero
import os

from gtorch_utils.constants import DB
from torch.utils.data import DataLoader

import settings
from consep.dataloaders.train_loader import FileLoader, SeedWorker
from consep.utils.patches.constants import PatchExtractType
from consep.utils.patches.patches import ProcessDataset
from utils.segmentation.plot import plot_img_and_mask


logzero.loglevel(settings.LOG_LEVEL)


def main():
    num_gpus = 1
    patch_size = (540, 540)
    step_size = (164, 164)
    model_input_shape = (270, 270)
    model_outut_shape = (270, 270)  # (80, 80)
    batch_size = 16  # train and val
    run_mode = DB.TRAIN
    num_workers = 0

    ###########################################################################
    #                      Extracting patches from CoNSeP                      #
    ###########################################################################
    db_info = {
        "train": {
            "img": (".png", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Train/Images/"),
            "ann": (".mat", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Train/Labels/"),
        },
        "valid": {
            "img": (".png", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Test/Images/"),
            "ann": (".mat", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Test/Labels/"),
        },
    }

    # ProcessDataset(dataset_info=db_info, win_size=patch_size,
    #                step_size=step_size, extract_type=PatchExtractType.MIRROR)()

    ###########################################################################
    #                          LOADING CoNSeP patches                         #
    ###########################################################################

    train_path = 'dataset/training_data/consep/train/540x540_164x164'
    train_list = glob.glob(os.path.join(train_path, '*.npy'))
    train_list.sort()

    # val_path = 'dataset/training_data/consep/valid/540x540_164x164'

    input_dataset = FileLoader(
        file_list=train_list,
        input_shape=model_input_shape,
        mask_shape=model_outut_shape,
        mode=DB.TRAIN,
        setup_augmentor=True,
    )

    train_dataloader = DataLoader(
        input_dataset,
        num_workers=num_workers,
        batch_size=batch_size * num_gpus,
        shuffle=run_mode == DB.TRAIN,
        drop_last=run_mode == DB.TRAIN,
        **SeedWorker(preserve_reproductibility=True)(),
    )

    data = next(iter(train_dataloader))

    for i in range(batch_size * num_gpus):
        plot_img_and_mask(data['img'][i, :], data['mask'][i, :])

    # __import__("pdb").set_trace()

    # TODO: something weird with the crops... regenerate then without augmentations /reflections
    # TODO: modify the data loader to save images and mask to disk!


if __name__ == '__main__':
    main()
