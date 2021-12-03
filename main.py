# -*- coding: utf-8 -*-
""" main """

import glob
import logzero
import os

import torch
from gtorch_utils.constants import DB
from gtorch_utils.nns.models.segmentation import UNet
from gtorch_utils.segmentation.visualisation import plot_img_and_mask
from torch.utils.data import DataLoader

import settings
from consep.dataloaders import OnlineCoNSePDataset, SeedWorker, OfflineCoNSePDataset
from consep.processors.offline import CreateDataset
from consep.utils.patches.constants import PatchExtractType
from consep.utils.patches.patches import ProcessDataset
from nns.managers import ModelMGR


logzero.loglevel(settings.LOG_LEVEL)


def main():
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

    # ProcessDataset(dataset_info=db_info, win_size=settings.PATCH_SIZE,
    #                step_size=settings.PATCH_STEP_SIZE, extract_type=PatchExtractType.MIRROR,
    #                type_classification=True, ann_percentage=.3)()

    ###########################################################################
    #                          LOADING CoNSeP patches                         #
    ###########################################################################

    # run_mode = DB.TRAIN
    # train_path = 'dataset/training_data/consep/train/540x540_164x164'
    # train_list = glob.glob(os.path.join(train_path, '*.npy'))
    # train_list.sort()

    # # val_path = 'dataset/training_data/consep/valid/540x540_164x164'

    # input_dataset = OnlineCoNSePDataset(
    #     file_list=train_list,
    #     crop_img_shape=settings.CROP_IMG_SHAPE,
    #     crop_mask_shape=settings.CROP_MASK_SHAPE,
    #     mode=DB.TRAIN,
    #     setup_augmentor=True,
    # )

    # train_dataloader = DataLoader(
    #     input_dataset,
    #     num_workers=settings.NUM_WORKERS,
    #     batch_size=settings.TOTAL_BATCH_SIZE,
    #     shuffle=run_mode == DB.TRAIN,
    #     drop_last=run_mode == DB.TRAIN,
    #     **SeedWorker(preserve_reproductibility=True)(),
    # )

    # data = next(iter(train_dataloader))

    # for i in range(settings.TOTAL_BATCH_SIZE):
    #     plot_img_and_mask(data['img'][i, :], data['mask'][i, :])

    ###########################################################################
    #                       CREATING THE OFFLINE DATASET                      #
    ###########################################################################

    # CreateDataset(
    #     train_path='dataset/training_data/consep/train/540x540_164x164',
    #     val_path='dataset/training_data/consep/valid/540x540_164x164',
    #     crop_img_shape=settings.CROP_IMG_SHAPE,
    #     crop_mask_shape=settings.CROP_MASK_SHAPE,
    #     num_gpus=settings.NUM_GPUS,
    #     num_workers=settings.NUM_WORKERS,
    #     saving_path=settings.CREATEDATASET_SAVING_PATH,
    # )()

    ###########################################################################
    #                      Loading ConSeP images dataset                       #
    ###########################################################################

    # run_mode = DB.TRAIN
    # train, val, test = OfflineCoNSePDataset.get_subdatasets(
    #     train_path=settings.CONSEP_TRAIN_PATH, val_path=settings.CONSEP_VAL_PATH)

    # train_dataloader = DataLoader(
    #     train,
    #     num_workers=settings.NUM_WORKERS,
    #     batch_size=settings.TOTAL_BATCH_SIZE,
    #     shuffle=run_mode == DB.TRAIN,
    #     drop_last=run_mode == DB.TRAIN,
    #     **SeedWorker(preserve_reproductibility=True)(),
    # )
    # data = next(iter(train_dataloader))

    ###########################################################################
    #                               Experiments                               #
    ###########################################################################
    # model = ModelMGR(
    #     model=torch.nn.DataParallel(UNet(n_channels=3, n_classes=10, bilinear=True)),
    #     cuda=True,
    #     epochs=10,
    #     intrain_val=2,
    #     optimizer=torch.optim.Adam,
    #     optimizer_kwargs=dict(lr=1e-3),
    #     labels_data=MyLabelClass,
    #     dataset=OfflineCoNSePDataset,
    #     dataset_kwargs={
    #         'train_path': settings.CONSEP_TRAIN_PATH,
    #         'val_path': settings.CONSEP_VAL_PATH,
    #         'test_path': settings.CONSEP_TEST_PATH,
    #     },
    #     train_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
    #     },
    #     testval_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
    #     },
    #     lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
    #     lr_scheduler_kwargs={'mode': 'max', 'patience': 2},  # {'step_size': 10, 'gamma': 0.1},
    #     criterions=[
    #         # nn.BCEWithLogitsLoss()
    #         # MSSSIMLoss(window_size=11, size_average=True, channel=3),
    #         # nn.BCELoss(size_average=True),
    #         # IOULoss(),
    #         # nn.CrossEntropyLoss()
    #         # bce_dice_loss  # 1866.6306
    #         # bce_dice_loss_ # 1890.8262
    #         DC_RNPV_LOSS(dice_threshold=0.25, always_conditioned=True)
    #     ],
    #     mask_threshold=0.5,
    #     metric=DC_RNPV(dice_threshold=0.25, always_conditioned=True),   # dice_coeff,
    #     earlystopping_kwargs=dict(min_delta=1e-3, patience=np.inf),
    #     checkpoint_interval=1,
    #     train_eval_chkpt=True,
    #     ini_checkpoint='',
    #     dir_checkpoints=settings.DIR_CHECKPOINTS,
    #     tensorboard=True,
    #     # TODO: there a bug that appeared once when plotting to disk after a long training
    #     # anyway I can always plot from the checkpoints :)
    #     plot_to_disk=False,
    #     plot_dir=settings.PLOT_DIRECTORY
    # )()


if __name__ == '__main__':
    main()
