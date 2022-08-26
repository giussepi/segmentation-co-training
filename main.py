# -*- coding: utf-8 -*-
""" main """

import glob
import os

import logzero
import matplotlib.pyplot as plt
import numpy as np
import torch
from gtorch_utils.constants import DB
from gtorch_utils.nns.models.segmentation import UNet, UNet_3Plus_DeepSup, UNet_3Plus, UNet_3Plus_DeepSup_CGM
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.segmentation import loss_functions
from gtorch_utils.segmentation.loss_functions.dice import dice_coef_loss
from gtorch_utils.segmentation.visualisation import plot_img_and_mask
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary

import settings
from consep.dataloaders import OnlineCoNSePDataset, SeedWorker, OfflineCoNSePDataset
from consep.datasets.constants import BinaryCoNSeP
from consep.processors.offline import CreateDataset
from consep.utils.patches.constants import PatchExtractType
from consep.utils.patches.patches import ProcessDataset
from ct82.datasets import CT82Dataset, CT82Labels
from ct82.images import NIfTI, ProNIfTI
from ct82.processors import CT82MGR
from ct82.settings import TRANSFORMS
from nns.backbones import resnet101, resnet152, xception
from nns.callbacks.metrics.constants import MetricEvaluatorMode
from nns.managers import ModelMGR, DAModelMGR
from nns.mixins.constants import LrShedulerTrack
from nns.models import Deeplabv3plus, UNet_3Plus_DA, UNet_3Plus_DA_Train, UNet_3Plus_DA2, \
    UNet_3Plus_DA2_Train, UNet_3Plus_DA2Ext, UNet_3Plus_DA2Ext_Train, AttentionUNet, AttentionUNet2, \
    UNet_3Plus_Intra_DA, UNet_3Plus_Intra_DA_GS, UNet_3Plus_Intra_DA_GS_HDX, XAttentionUNet, UNet2D, \
    UNet_Grid_Attention, UNet_Att_DSV, SingleAttentionBlock, \
    MultiAttentionBlock, UNet3D, XGridAttentionUNet, XAttentionUNet_DSV2
from nns.models.layers.disagreement_attention import inter_model
from nns.models.layers.disagreement_attention import intra_model
from nns.models.layers.disagreement_attention.constants import AttentionMergingType
from nns.segmentation.learning_algorithms import CoTraining, DACoTraining
from nns.segmentation.utils.postprocessing import ExpandPrediction
from nns.utils.sync_batchnorm import get_batchnormxd_class


logzero.loglevel(settings.LOG_LEVEL)


def bce_dice_loss(inputs, target):
    """
    Same as bce_dice_loss_ but this works a bit faster

    bce_dice_loss   1866.6306 s
    bce_dice_loss_  1890.8262 s

    Source: https://www.kaggle.com/bonhart/brain-tumor-multi-class-segmentation-baseline

    Returns:
      dice_loss + bce_loss
    """
    dice_loss = dice_coef_loss(inputs, target)
    bceloss = torch.nn.BCEWithLogitsLoss()(inputs, target)

    return bceloss + dice_loss


class BceDiceLoss(torch.nn.Module):
    """
    Module based BceDiceLoss with logits support

    Usage:
        BceDiceLoss()(predictions, ground_truth)
    """

    def __init__(self, *, with_logits=False):
        """ Initializes the object instance """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)

        self.with_logits = with_logits

    def forward(self, preds, targets):
        """
        Calculates and returns the bce_dice loss

        Kwargs:
            preds  <torch.Tensor>: predicted masks [batch_size,  ...]
            target <torch.Tensor>: ground truth masks [batch_size, ...]

        Returns:
            loss <torch.Tensor>
        """
        if self.with_logits:
            return bce_dice_loss(preds, targets)

        return bce_dice_loss(preds, targets)


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
    model1 = dict(
        # model=torch.nn.DataParallel(UNet_3Plus_DeepSup_CGM(n_channels=3, n_classes=1, is_deconv=False)),
        # model=torch.nn.DataParallel(UNet_3Plus_DeepSup(n_channels=3, n_classes=1, is_deconv=False)),
        model=UNet_3Plus,
        model_kwargs=dict(n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.XAVIER,
                          batchnorm_cls=get_batchnormxd_class()),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=30,  # 20
        intrain_val=2,  # 2
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-4),  # lr=1e-3
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
        lr_scheduler_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler_track=LrShedulerTrack.LOSS,
        criterions=[
            # torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
            loss_functions.BceDiceLoss(with_logits=True),
            loss_functions.SpecificityLoss(with_logits=True),
        ],
        mask_threshold=0.5,
        metrics=settings.METRICS,
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
        checkpoint_interval=0,
        train_eval_chkpt=False,
        last_checkpoint=True,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'unet3_plus_1'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # model1.print_data_logger_summary()

    # model2 = ModelMGR(
    model2 = dict(
        # model=torch.nn.DataParallel(UNet_3Plus_DeepSup_CGM(n_channels=3, n_classes=1, is_deconv=False)),
        # model=torch.nn.DataParallel(UNet_3Plus_DeepSup(n_channels=3, n_classes=1, is_deconv=False)),
        model=UNet_3Plus,
        model_kwargs=dict(n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.KAIMING,
                          batchnorm_cls=get_batchnormxd_class()),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=30,  # 20
        intrain_val=2,  # 2
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-4),  # lr=1e-3
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
        lr_scheduler_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler_track=LrShedulerTrack.LOSS,
        criterions=[
            # torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
            loss_functions.BceDiceLoss(with_logits=True),
            loss_functions.SpecificityLoss(with_logits=True),
        ],
        mask_threshold=0.5,
        metrics=settings.METRICS,
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
        checkpoint_interval=0,
        train_eval_chkpt=False,
        last_checkpoint=True,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'unet3_plus_2'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # model2()
    # model2.predict('1.ann.tiff', Image.open, patch_size=256, patch_overlapping=2, superimpose=False, size=None)

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
            batchnorm=get_batchnormxd_class(), backbone=xception, backbone_pretrained=True,
            dilated=True, multi_grid=False, deep_base=True
        ),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=30,  # 20
        intrain_val=2,  # 2
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-4),  # 1e-3, try 1e-4, weight_decay=4e-5
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
        lr_scheduler_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler_track=LrShedulerTrack.LOSS,
        criterions=[
            # torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
            loss_functions.BceDiceLoss(with_logits=True),
            loss_functions.SpecificityLoss(with_logits=True),
        ],
        mask_threshold=0.5,
        metrics=settings.METRICS,
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
        checkpoint_interval=0,
        train_eval_chkpt=False,
        last_checkpoint=True,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(
            settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'deeplabv3plus_xception'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # model3.predict('1.ann.tiff', Image.open, patch_size=256, patch_overlapping=2, superimpose=False, size=None)
    # model3()
    # model3.print_data_logger_summary()
    # model3.plot_and_save(None, 154)

    # cot = CoTraining(
    #     model_mgr_kwargs_list=[model1, model2],
    #     iterations=5,
    #     # model_mgr_kwargs_tweaks=[
    #     #     dict(optimizer_kwargs=dict(lr=1e-3), lr_scheduler_kwargs={'mode': 'min', 'patience': 1}),
    #     #     dict(optimizer_kwargs=dict(lr=1e-3), lr_scheduler_kwargs={'mode': 'min', 'patience': 1})
    #     # ],
    #     metrics=settings.METRICS,
    #     earlystopping_kwargs=dict(min_delta=1e-3, patience=2),
    #     warm_start=None,  # dict(lamda=.0, sigma=.0),  # dict(lamda=.5, sigma=.01),
    #     overall_best_models=False,  # True
    #     dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109'),
    #     # thresholds=dict(agreement=.65, disagreement=(.25, .7)),  # dict(agreement=.8, disagreement=(.25, .8))
    #     thresholds=dict(disagreement=(.25, .8)),
    #     plots_saving_path=settings.PLOT_DIRECTORY,
    #     strategy_postprocessing=dict(
    #         disagreement=[ExpandPrediction(), ],
    #     ),
    #     general_postprocessing=[],
    #     postprocessing_threshold=.8,  # try same experiment but raising the thresholds to reduce errors
    #     dataset=OfflineCoNSePDataset,
    #     dataset_kwargs={
    #         'train_path': settings.CONSEP_TRAIN_PATH,
    #         'val_path': settings.CONSEP_VAL_PATH,
    #         'test_path': settings.CONSEP_TEST_PATH,
    #         'cotraining': settings.COTRAINING,
    #         'original_masks': settings.ORIGINAL_MASKS,
    #     },
    #     train_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
    #     },
    #     testval_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
    #     }
    # )
    # cot()

    # try:
    #     cot.print_data_logger_summary(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'chkpt_4.pth.tar'))

    #     cot.plot_and_save(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'chkpt_4.pth.tar'),
    #         save=True, show=False, dpi=300.
    #     )

    #     cot.print_data_logger_details(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'chkpt_4.pth.tar'))
    # except:
    #     pass

    # DA experiments ##########################################################
    # TODO: update doctrings from DAModelMGRMixin
    # model4 = DAModelMGR(
    model4 = dict(
        model_cls=UNet_3Plus_DA_Train,
        model_kwargs=dict(
            model1_cls=UNet_3Plus_DA,
            kwargs1=dict(da_threshold=np.NINF, da_block_cls=inter_model.ThresholdedDisagreementAttentionBlock,
                         da_block_config=dict(thresholds=(.25, .8), beta=0.),
                         # da_merging_type=AttentionMergingType.MAX,
                         n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.XAVIER,
                         batchnorm_cls=get_batchnormxd_class()),
            model2_cls=UNet_3Plus_DA,
            kwargs2=dict(da_threshold=np.NINF, da_block_cls=inter_model.ThresholdedDisagreementAttentionBlock,
                         da_block_config=dict(thresholds=(.25, .8), beta=0.),
                         # da_merging_type=AttentionMergingType.MAX,
                         n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.KAIMING,
                         batchnorm_cls=get_batchnormxd_class()),
        ),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=30,
        intrain_val=2,
        optimizer1=torch.optim.Adam,
        optimizer1_kwargs=dict(lr=1e-4),  # lr=1e-3
        optimizer2=torch.optim.Adam,
        optimizer2_kwargs=dict(lr=1e-4),  # lr=1e-3
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
        lr_scheduler1=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
        # TODO: the mode can change based on the quantity monitored
        # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        lr_scheduler1_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler1_track=LrShedulerTrack.LOSS,
        lr_scheduler2=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
        lr_scheduler2_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
        lr_scheduler2_track=LrShedulerTrack.LOSS,
        criterions=[
            # torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
            loss_functions.BceDiceLoss(with_logits=True),
            # BceDiceLoss(),
            loss_functions.SpecificityLoss(with_logits=True),
        ],
        mask_threshold=0.5,
        metrics=settings.METRICS,
        metric_mode=MetricEvaluatorMode.MAX,
        process_joint_values=False,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
        checkpoint_interval=0,
        train_eval_chkpt=False,
        last_checkpoint=True,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'unet3_plus_DA'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # summary(model4.module, depth=10, verbose=1)
    # model4()
    # model4.predict('1.ann.tiff', Image.open, patch_size=256, patch_overlapping=2, superimpose=False, size=None)
    # model4.print_data_logger_summary()
    # _, data_logger = model4.load_checkpoint([
    #     model4.optimizer1(model4.module.model1.parameters(), **model4.optimizer1_kwargs),
    #     model4.optimizer2(model4.module.model2.parameters(), **model4.optimizer2_kwargs),
    # ])
    # model4.plot_and_save(152)

    # model5 = dict(
    # model5 = ModelMGR(
    #     # model=torch.nn.DataParallel(UNet_3Plus_DeepSup_CGM(n_channels=3, n_classes=1, is_deconv=False)),
    #     # model=torch.nn.DataParallel(UNet_3Plus_DeepSup(n_channels=3, n_classes=1, is_deconv=False)),
    #     model=AttentionUNet2,
    #     model_kwargs=dict(n_channels=3, n_classes=1, batchnorm_cls=get_batchnormxd_class()),
    #     cuda=settings.CUDA,
    #     multigpus=settings.MULTIGPUS,
    #     patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
    #     epochs=30,  # 20
    #     intrain_val=2,  # 2
    #     optimizer=torch.optim.Adam,
    #     optimizer_kwargs=dict(lr=1e-4),  # lr=1e-3
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
    #     lr_scheduler_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
    #     lr_scheduler_track=LrShedulerTrack.LOSS,
    #     criterions=[
    #         # torch.nn.BCEWithLogitsLoss()
    #         # torch.nn.CrossEntropyLoss()
    #         loss_functions.BceDiceLoss(with_logits=True),
    #         # BceDiceLoss(),
    #         loss_functions.SpecificityLoss(with_logits=True),
    #     ],
    #     mask_threshold=0.5,
    #     metrics=settings.METRICS,
    #     metric_mode=MetricEvaluatorMode.MAX,
    #     earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
    #     checkpoint_interval=0,
    #     train_eval_chkpt=False,
    #     last_checkpoint=True,
    #     ini_checkpoint='',
    #     dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'unet3_plus_2'),
    #     tensorboard=False,
    #     # TODO: there a bug that appeared once when plotting to disk after a long training
    #     # anyway I can always plot from the checkpoints :)
    #     plot_to_disk=False,
    #     plot_dir=settings.PLOT_DIRECTORY
    # )
    # model5()
    # model5.predict('1.ann.tiff', Image.open, patch_size=256, patch_overlapping=2, superimpose=False, size=None)
    # model5.print_data_logger_summary()
    # model5.plot_and_save(None, 154)

    # model6 = dict(
    # model6 = ModelMGR(
    #     model=XAttentionUNet_DSV2,  # XAttentionUNet,  # UNet_Att_DSV,  # UNet2D,  # UNet_Grid_Attention,  # AttentionUNet2, # UNet_3Plus,
    #     model_kwargs=dict(da_block_cls=intra_model.AttentionBlock,
    #                       # da_block_config=dict(thresholds=(.25, .8), beta=.4, n_channels=-1),
    #                       # da_block_config=dict(n_channels=-1),
    #                       # is_deconv=True,
    #                       # feature_scale=1, is_batchnorm=True,
    #                       bilinear=False,  # XAttentionUNet only
    #                       n_channels=3, n_classes=1,
    #                       init_type=UNet3InitMethod.KAIMING,
    #                       data_dimensions=settings.DATA_DIMENSIONS,
    #                       batchnorm_cls=get_batchnormxd_class(),
    #                       dsv=True,
    #                       ),
    #     cuda=settings.CUDA,
    #     multigpus=settings.MULTIGPUS,
    #     patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
    #     epochs=20,  # 30
    #     intrain_val=2,  # 2
    #     optimizer=torch.optim.Adam,
    #     optimizer_kwargs=dict(lr=1e-4),  # lr=1e-3
    #     sanity_checks=False,
    #     labels_data=BinaryCoNSeP,
    #     data_dimensions=settings.DATA_DIMENSIONS,
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
    #     lr_scheduler_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
    #     lr_scheduler_track=LrShedulerTrack.LOSS,
    #     criterions=[
    #         # torch.nn.BCEWithLogitsLoss()
    #         # torch.nn.CrossEntropyLoss()
    #         loss_functions.BceDiceLoss(with_logits=True),
    #         # BceDiceLoss(),
    #         # loss_functions.SpecificityLoss(with_logits=True), # now it's throwing an exception :/
    #     ],
    #     mask_threshold=0.5,
    #     metrics=settings.METRICS,
    #     metric_mode=MetricEvaluatorMode.MAX,
    #     earlystopping_kwargs=dict(min_delta=1e-3, patience=10, metric=True),
    #     checkpoint_interval=0,
    #     train_eval_chkpt=False,
    #     last_checkpoint=True,
    #     ini_checkpoint='',
    #     dir_checkpoints=os.path.join(
    #         settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'unet_3plus_intra_da'),
    #     tensorboard=False,
    #     # TODO: there a bug that appeared once when plotting to disk after a long training
    #     # anyway I can always plot from the checkpoints :)
    #     plot_to_disk=False,
    #     plot_dir=settings.PLOT_DIRECTORY
    # )
    # model6()
    # model6.print_data_logger_summary()
    # model6.plot_and_save(None, 154)
    # summary(model6.module, (4, 3, *settings.CROP_IMG_SHAPE), depth=1, verbose=1)

    ##
    ###########################################################################
    #                           Working with 3D data                          #
    ###########################################################################
    # m = UNet3D(feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True)
    model7 = ModelMGR(
        model=XAttentionUNet_DSV2,  # UNet_Att_DSV,  # UNet_Grid_Attention,,  # XAttentionUNet,  # UNet3D,
        # UNet3D
        # model_kwargs=dict(feature_scale=1, n_channels=1, n_classes=1, is_batchnorm=True),
        # XAttentionUNet & XGridAttentionUNet
        model_kwargs=dict(
            n_channels=1, n_classes=1, bilinear=False,
            batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
            data_dimensions=settings.DATA_DIMENSIONS, da_block_cls=intra_model.MixedEmbeddedDABlock,
            dsv=True
        ),
        # UNet_Att_DSV
        # model_kwargs=dict(feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True,
        #                   attention_block_cls=SingleAttentionBlock, data_dimensions=settings.DATA_DIMENSIONS),
        # UNet_Grid_Attention
        # model_kwargs=dict(feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True,
        #                   data_dimensions=settings.DATA_DIMENSIONS),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=1000,  # 1000
        intrain_val=2,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6),
        sanity_checks=False,
        labels_data=CT82Labels,
        data_dimensions=settings.DATA_DIMENSIONS,
        dataset=CT82Dataset,
        dataset_kwargs={
            'train_path': settings.CT82_TRAIN_PATH,
            'val_path': settings.CT82_VAL_PATH,
            'test_path': settings.CT82_TEST_PATH,
            'cotraining': settings.COTRAINING,
            'cache': settings.DB_CACHE,
        },
        train_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
        },
        testval_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
        },
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={'step_size': 250, 'gamma': 0.5},
        # TODO: the mode can change based on the quantity monitored
        # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        lr_scheduler_track=LrShedulerTrack.NO_ARGS,
        criterions=[
            # torch.nn.BCEWithLogitsLoss()
            # torch.nn.CrossEntropyLoss()
            loss_functions.BceDiceLoss(with_logits=True),
            # BceDiceLoss(),
            # loss_functions.SpecificityLoss(with_logits=True),
        ],
        mask_threshold=0.5,
        metrics=settings.METRICS,
        metric_mode=MetricEvaluatorMode.MAX,
        earlystopping_kwargs=dict(min_delta=1e-3, patience=np.inf, metric=True),  # patience=10
        checkpoint_interval=0,
        train_eval_chkpt=False,
        last_checkpoint=True,
        ini_checkpoint='',
        dir_checkpoints=os.path.join(
            settings.DIR_CHECKPOINTS, 'ct82',  'unet3d', 'exp1'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    model7()
    # model7.print_data_logger_summary()
    # model7.plot_and_save(None, 154)
    # summary(model6.module, (4, 3, *settings.CROP_IMG_SHAPE), depth=1, verbose=1)
    # m = UNet3D(feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True)
    # summary(model7.module, (1, 1, *settings.CT82_CROP_SHAPE), depth=1, verbose=1)
    ##

    ###########################################################################
    # Disagreement attention cotraining experiments ###########################
    # cot = DACoTraining(
    #     model_mgr_kwargs=model4,
    #     iterations=5,
    #     # model_mgr_kwargs_tweaks=dict(
    #     #     optimizer1_kwargs=dict(lr=1e-3),
    #     #     optimizer2_kwargs=dict(lr=1e-3),
    #     #     lr_scheduler1_kwargs={'mode': 'min', 'patience': 1},
    #     #     lr_scheduler2_kwargs={'mode': 'min', 'patience': 1},
    #     # ),
    #     metrics=settings.METRICS,
    #     earlystopping_kwargs=dict(min_delta=1e-3, patience=2),
    #     warm_start=None,  # dict(lamda=.0, sigma=.0),  # dict(lamda=.5, sigma=.01),
    #     overall_best_models=True,
    #     dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109'),
    #     # thresholds=dict(agreement=.65, disagreement=(.25, .7)),  # dict(agreement=.8, disagreement=(.25, .8))
    #     thresholds=dict(agreement=.8),
    #     plots_saving_path=settings.PLOT_DIRECTORY,
    #     strategy_postprocessing=dict(
    #         # disagreement=[ExpandPrediction(), ],
    #     ),
    #     general_postprocessing=[],
    #     postprocessing_threshold=.8,
    #     dataset=OfflineCoNSePDataset,
    #     dataset_kwargs={
    #         'train_path': settings.CONSEP_TRAIN_PATH,
    #         'val_path': settings.CONSEP_VAL_PATH,
    #         'test_path': settings.CONSEP_TEST_PATH,
    #         'cotraining': settings.COTRAINING,
    #         'original_masks': settings.ORIGINAL_MASKS,
    #     },
    #     train_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
    #     },
    #     testval_dataloader_kwargs={
    #         'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
    #     }
    # )
    # cot()

    # try:
    #     cot.print_data_logger_summary(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'chkpt_4.pth.tar'))

    #     cot.plot_and_save(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'chkpt_4.pth.tar'),
    #         save=True, show=False, dpi=300.
    #     )

    #     cot.print_data_logger_details(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp109', 'chkpt_4.pth.tar'))
    # except Exception:
    #     pass

    ###############################################################################
    #                                CT-82 dataset                                #
    ###############################################################################

    # # Processing CT-82 dataset ################################################
    # target_size = (368, 368, 96)
    # target_size = (368, 368, -1)
    # target_size = (368, 368, 145)
    # mgr = CT82MGR(target_size=target_size)
    # mgr()

    # assert len(glob.glob(os.path.join(mgr.saving_labels_folder, r'*.nii.gz'))) == 80
    # assert len(glob.glob(os.path.join(mgr.saving_cts_folder, r'*.pro.nii.gz'))) == 80

    # files_idx = [*range(1, 83)]
    # for id_ in mgr.non_existing_ct_folders[::-1]:
    #     files_idx.pop(id_-1)

    # for subject in files_idx:
    #     labels = NIfTI(os.path.join(mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
    #     cts = ProNIfTI(os.path.join(mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
    #     assert labels.shape == cts.shape == target_size

    # mgr.perform_visual_verification(80, scans=[70], clahe=True)
    # mgr.split_processed_dataset(.15, .25, shuffle=False)

    # visual verification of cts ##############################################
    # target_size = (368, 368, 96)  # (1024, 1024, 96)
    # mgr = CT82MGR(
    #     saving_path='CT-82-Pro',
    #     target_size=target_size
    # )
    # mgr.non_existing_ct_folders = []
    # mgr.perform_visual_verification(1, scans=[72], clahe=True)
    # # os.remove(mgr.VERIFICATION_IMG)

    # getting subdatasets and plotting some crops #############################
    # train, val, test = CT82Dataset.get_subdatasets()
    # for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    #     print(f'{db_name}: {len(dataset)}')
    #     data = dataset[0]
    #     print(data['image'].shape, data['mask'].shape)
    #     print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])

    #     print(data['image'].min(), data['image'].max())
    #     print(data['mask'].min(), data['mask'].max())

    #     img_id = np.random.randint(0, 72)
    #     if len(data['image'].shape) == 4:
    #         fig, axis = plt.subplots(1, 2)
    #         axis[0].imshow(data['image'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()
    #     else:
    #         fig, axis = plt.subplots(2, 4)
    #         for idx, i, m in zip([*range(4)], data['image'], data['mask']):
    #             axis[0, idx].imshow(i.detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #             axis[1, idx].imshow(m.detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()
    ###############################################################################
    #                                CT-82 dataset                                #
    ###############################################################################
    # print(CT82MGR().get_insights())
    # DICOM files 18942
    # NIfTI labels 82
    # MIN_VAL = -2048
    # MAX_VAL = 3071
    # MIN_NIFTI_SLICES_WITH_DATA = 46
    # MAX_NIFTI_SLICES_WITH_DATA = 145
    # folders PANCREAS_0025 and PANCREAS_0070 are empty
    # MIN DICOMS per subject 181
    # MAX DICOMS per subject 466
    ###############################################################################
    #                                     dpis                                    #
    # https://www.iprintfromhome.com/mso/understandingdpi.pdf very good explanantion of dpi
    # dpi = dim px / size inches
    # width = height = 512px
    # print size = 7.1111111... inches
    # dpi = 72 / 25.4 = 2.8346 pixel/mm

    # target = isotropic 2.00 pixel/mm  = 2*25.4 = 50.8 dpi or pixel/in
    # print size = 7.111111.. inches
    # widh = height = 50.8 * 7.11111 = 361.2443 px

    # 361.2443 % 16 != 0 so to avoid any padding or resize when using UNet
    # we can use 352x352[49.50 dpi or 1.9488 px/in] or 368x368 [51.75 dpi 2.0374 px/in]

    # using width = height = 160
    # dpi = 22.500 = 0.8858 px/in

    ###############################################################################
    ###############################################################################
    #                                 UNEt details                                #
    # 3D model
    # small batches 2 - 4
    # standard data-augmentation techniques (affine transformations, axial flips, random crops)
    # Intensity values are linearly scaled to obtain a normal distribution N (0, 1)
    # https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/configs/config_unet_ct_dsv.json
    # lr_scheduler https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/networks_other.py#L101
    #
    # Sorensen-Dice loss https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/layers/loss.py#L29
    # Adam https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/utils.py#L23
    # transforms https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/dataio/transformation/transforms.py#L77
    # CT-150 train 120 testing 30
    # The results on pancreas predictions demonstrate that attention gates (AGs)
    # increase recall values (p = .005) by improving the model’s expression power as it relies
    # on AGs to localise foreground pixels.
    # inference timeused 160x160x96 tensors
    # CT-82 (TCIA Pancreas-CT Dataset) train 61 (74.39%), test 21 (25.6%)
    # 5-fold cross-validation
    # #+caption: models from scratch
    # | Method          | Dice        | Precision   | Recall      | S2S dist(mm) |
    # |-----------------+-------------+-------------+-------------+--------------|
    # | U-Net [24]      | 0.815±0.068 | 0.815±0.105 | 0.826±0.062 | 2.576±1.180  |
    # | Attention U-Net | 0.821±0.057 | 0.815±0.093 | 0.835±0.057 | 2.333±0.856  |
    # model config https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/configs/config_unet_ct_dsv.json
    # "augmentation": {
    #     "acdc_sax": {
    #       "shift": [0.1,0.1],
    #       "rotate": 15.0,
    #       "scale": [0.7,1.3],
    #       "intensity": [1.0,1.0],
    #       "random_flip_prob": 0.5,
    #       "scale_size": [160,160,96],
    #       "patch_size": [160,160,96]
    #     }
    #   },
    # epochs >= 150 (1000)
    ###############################################################################
    ###############################################################################
    #                                    DICOM                                    #
    # https://towardsdatascience.com/understanding-dicoms-835cd2e57d0b
    # 16 bit DICOM images have values ranging from -32768 to 32768 while 8-bit grey-scale images
    # store values from 0 to 255.
    ###############################################################################

    ##
    # Affine transformations examples: translation, scaling, homothety, similarity, reflection,
    # rotation, shear mapping and compositions of them in any combination sequence
    ##


if __name__ == '__main__':
    main()
