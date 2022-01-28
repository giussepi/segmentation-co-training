# -*- coding: utf-8 -*-
""" main """

import glob
import os

import logzero
import numpy as np
import torch
from gtorch_utils.constants import DB
from gtorch_utils.nns.models.segmentation import UNet, UNet_3Plus_DeepSup, UNet_3Plus, UNet_3Plus_DeepSup_CGM
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.segmentation import metrics
from gtorch_utils.segmentation.visualisation import plot_img_and_mask
from torch.utils.data import DataLoader

import settings
from consep.dataloaders import OnlineCoNSePDataset, SeedWorker, OfflineCoNSePDataset
from consep.datasets.constants import BinaryCoNSeP
from consep.processors.offline import CreateDataset
from consep.utils.patches.constants import PatchExtractType
from consep.utils.patches.patches import ProcessDataset
from nns.backbones import resnet101, resnet152, xception
from nns.callbacks.metrics.constants import MetricEvaluatorMode
from nns.managers import ModelMGR
from nns.mixins.constants import LrShedulerTrack
from nns.models import Deeplabv3plus
from nns.segmentation.learning_algorithms import CoTraining
from nns.segmentation.utils.postprocessing import ExpandPrediction
from nns.utils.sync_batchnorm import get_batchnorm2d_class


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
    #                type_classification=True, ann_percentage=.7)()

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
    #     train_path='dataset_.7/training_data/consep/train/540x540_164x164',
    #     val_path='dataset_.7/training_data/consep/valid/540x540_164x164',
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
    # model1 = ModelMGR(
    # model1 = dict(
    #     # model=torch.nn.DataParallel(UNet_3Plus_DeepSup_CGM(n_channels=3, n_classes=1, is_deconv=False)),
    #     # model=torch.nn.DataParallel(UNet_3Plus_DeepSup(n_channels=3, n_classes=1, is_deconv=False)),
    #     model=torch.nn.DataParallel(UNet_3Plus(n_channels=3, n_classes=1,
    #                                            is_deconv=False, init_type=UNet3InitMethod.XAVIER)),
    #     # model=torch.nn.DataParallel(UNet(n_channels=3, n_classes=1, bilinear=True)),
    #     # model=UNet(n_channels=3, n_classes=1, bilinear=True),
    #     # logits=True, # TODO: review if it is still necessary
    #     # sigmoid=False, # TODO: review if it is still necessary
    #     cuda=True,
    #     epochs=12,  # 20
    #     intrain_val=2,  # 2
    #     optimizer=torch.optim.Adam,
    #     optimizer_kwargs=dict(lr=1e-3),
    #     labels_data=BinaryCoNSeP,
    #     dataset=OfflineCoNSePDataset,
    #     dataset_kwargs={
    #         'train_path': settings.CONSEP_TRAIN_PATH,
    #         'val_path': settings.CONSEP_VAL_PATH,
    #         'test_path': settings.CONSEP_TEST_PATH,
    #         'cotraining': settings.COTRAINING,
    #     },
    #     train_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
    #     },
    #     testval_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
    #     },
    #     lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
    #     # TODO: the mode can change based on the quantity monitored
    #     # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
    #     lr_scheduler_kwargs={'mode': 'min', 'patience': 2},  # {'step_size': 10, 'gamma': 0.1},
    #     lr_scheduler_track=LrShedulerTrack.LOSS,
    #     criterions=[
    #         torch.nn.BCEWithLogitsLoss()
    #         # torch.nn.CrossEntropyLoss()
    #     ],
    #     mask_threshold=0.5,
    #     metric=metrics.dice_coeff_metric,
    #     metric_mode=MetricEvaluatorMode.MAX,
    #     earlystopping_kwargs=dict(min_delta=1e-3, patience=7, metric=True),
    #     checkpoint_interval=1,
    #     train_eval_chkpt=False,
    #     ini_checkpoint='',
    #     dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp29', 'model1'),
    #     tensorboard=False,
    #     # TODO: there a bug that appeared once when plotting to disk after a long training
    #     # anyway I can always plot from the checkpoints :)
    #     plot_to_disk=False,
    #     plot_dir=settings.PLOT_DIRECTORY
    # )

    # model1.print_data_logger_summary()

    # model2 = ModelMGR(
    model2 = dict(
        # model=torch.nn.DataParallel(UNet_3Plus_DeepSup_CGM(n_channels=3, n_classes=1, is_deconv=False)),
        # model=torch.nn.DataParallel(UNet_3Plus_DeepSup(n_channels=3, n_classes=1, is_deconv=False)),
        model=UNet_3Plus,
        model_kwargs=dict(n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.KAIMING,
                          batchnorm_cls=get_batchnorm2d_class()),
        # logits=True, # TODO: review if it is still necessary
        # sigmoid=False, # TODO: review if it is still necessary
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=20,  # 20
        intrain_val=2,  # 2
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-3),
        labels_data=BinaryCoNSeP,
        dataset=OfflineCoNSePDataset,
        dataset_kwargs={
            'train_path': settings.CONSEP_TRAIN_PATH,
            'val_path': settings.CONSEP_VAL_PATH,
            'test_path': settings.CONSEP_TEST_PATH,
            'cotraining': settings.COTRAINING,
        },
        train_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
        },
        testval_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
        },
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
        # TODO: the mode can change based on the quantity monitored
        # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        lr_scheduler_kwargs={'mode': 'min', 'patience': 2},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler_track=LrShedulerTrack.LOSS,
        criterions=[
            torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
        ],
        mask_threshold=0.5,
        metric=metrics.dice_coeff_metric,
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
        checkpoint_interval=0,
        train_eval_chkpt=False,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp43', 'unet3_plus'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # # model2()

    # model3 = ModelMGR(
    model3 = dict(
        model=Deeplabv3plus,
        model_kwargs=dict(
            cfg=dict(model_aspp_outdim=256,
                     train_bn_mom=3e-4,
                     model_aspp_hasglobal=True,
                     model_shortcut_dim=48,
                     model_num_classes=1,
                     model_freezebn=False,
                     model_channels=3),
            batchnorm=get_batchnorm2d_class(), backbone=xception, backbone_pretrained=True,
            dilated=True, multi_grid=False, deep_base=True
        ),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=20,  # 20
        intrain_val=2,  # 2
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-3),
        labels_data=BinaryCoNSeP,
        dataset=OfflineCoNSePDataset,
        dataset_kwargs={
            'train_path': settings.CONSEP_TRAIN_PATH,
            'val_path': settings.CONSEP_VAL_PATH,
            'test_path': settings.CONSEP_TEST_PATH,
            'cotraining': settings.COTRAINING,
        },
        train_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
        },
        testval_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
        },
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
        # TODO: the mode can change based on the quantity monitored
        # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        lr_scheduler_kwargs={'mode': 'min', 'patience': 2},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler_track=LrShedulerTrack.LOSS,
        criterions=[
            torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
        ],
        mask_threshold=0.5,
        metric=metrics.dice_coeff_metric,
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
        checkpoint_interval=0,
        train_eval_chkpt=False,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(
            settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp43', 'deeplabv3plus_xception'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # model3()
    # model3.print_data_logger_summary()

    # RuntimeError: [enforce fail at inline_container.cc:300] . unexpected pos 596530496 vs 596530392

    cot = CoTraining(
        model_mgr_kwargs_list=[model2, model3],
        iterations=5,
        metric=metrics.dice_coeff_metric,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=2),
        warm_start=dict(lamda=.0, sigma=.0),  # dict(lamda=.5, sigma=.01),
        overall_best_models=False,
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp43'),
        thresholds=dict(disagreement=(.25, .8)),  # dict(agreement=.65, disagreement=(.25, .7)),
        plots_saving_path=settings.PLOT_DIRECTORY,
        mask_postprocessing=[
            ExpandPrediction(),
        ],
        postprocessing_threshold=.5,
        dataset=OfflineCoNSePDataset,
        dataset_kwargs={
            'train_path': settings.CONSEP_TRAIN_PATH,
            'val_path': settings.CONSEP_VAL_PATH,
            'test_path': settings.CONSEP_TEST_PATH,
            'cotraining': settings.COTRAINING,
        },
        train_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
        },
        testval_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
        }
    )
    cot()

    # cot.print_data_logger_summary(
    #     os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp43', 'chkpt_2.pth.tar'))

    # cot.plot_and_save(
    #     os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp43', 'chkpt_2.pth.tar'),
    #     save=True, show=False, dpi=300.
    # )


if __name__ == '__main__':
    main()
