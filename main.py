# -*- coding: utf-8 -*-
""" main """

import glob
import os
import re
import shutil
from collections import defaultdict

import logzero
import matplotlib.pyplot as plt
import numpy as np
import torch
from gtorch_utils.constants import DB
from gtorch_utils.datasets.segmentation.datasets.consep.dataloaders import OnlineCoNSePDataset, \
    SeedWorker, OfflineCoNSePDataset
from gtorch_utils.datasets.segmentation.datasets.consep.datasets.constants import BinaryCoNSeP
from gtorch_utils.datasets.segmentation.datasets.consep.processors.offline import CreateDataset
from gtorch_utils.datasets.segmentation.datasets.consep.utils.patches.constants import PatchExtractType
from gtorch_utils.datasets.segmentation.datasets.consep.utils.patches.patches import ProcessDataset
from gtorch_utils.datasets.segmentation.datasets.ct82.datasets import CT82Dataset, CT82Labels
from gtorch_utils.datasets.segmentation.datasets.ct82.processors import CT82MGR
from gtorch_utils.datasets.segmentation.datasets.ct82.settings import TRANSFORMS
from gtorch_utils.datasets.segmentation.datasets.lits17.processors import LiTS17MGR, LiTS17CropMGR
from gtorch_utils.datasets.segmentation.datasets.lits17.datasets import LiTS17OnlyLiverLabels, \
    LiTS17Dataset, LiTS17OnlyLesionLabels, LiTS17CropDataset
from gtorch_utils.nns.managers.callbacks.metrics.constants import MetricEvaluatorMode
from gtorch_utils.nns.mixins.constants import LrShedulerTrack
from gtorch_utils.nns.mixins.images_types import CT3DNIfTIMixin
from gtorch_utils.nns.models.backbones import resnet101, resnet152, xception
from gtorch_utils.nns.models.segmentation import UNet, UNet_3Plus_DeepSup, UNet_3Plus, UNet_3Plus_DeepSup_CGM, \
    Deeplabv3plus
from gtorch_utils.nns.models.segmentation.unet.unet_parts import TinyUpAE, TinyAE, MicroUpAE, MicroAE
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.nns.utils.sync_batchnorm import get_batchnormxd_class
# from gtorch_utils.nns.utils.reproducibility import Reproducibility
from gtorch_utils.segmentation import loss_functions
from gtorch_utils.segmentation.loss_functions.dice import dice_coef_loss
from gtorch_utils.segmentation.visualisation import plot_img_and_mask
from gutils.images.images import NIfTI, ProNIfTI
from monai.transforms import ForegroundMask
from PIL import Image
from skimage.exposure import equalize_adapthist
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import settings
from nns.managers import ModelMGR, DAModelMGR, ModularModelMGR, MultiPredsModelMGR, AEsModelMGR, \
    ADSVModelMGR, SDSVModelMGR
from nns.models import UNet_3Plus_DA, UNet_3Plus_DA_Train, UNet_3Plus_DA2, \
    UNet_3Plus_DA2_Train, UNet_3Plus_DA2Ext, UNet_3Plus_DA2Ext_Train, AttentionUNet, AttentionUNet2, \
    UNet_3Plus_Intra_DA, UNet_3Plus_Intra_DA_GS, UNet_3Plus_Intra_DA_GS_HDX, XAttentionUNet, UNet2D, \
    UNet_Grid_Attention, UNet_Att_DSV, SingleAttentionBlock, \
    MultiAttentionBlock, UNet3D, XGridAttentionUNet, UNet4Plus, ModularUNet4Plus, XAttentionAENet, \
    XAttentionUNet_ADSV, XAttentionUNet_SDSV
from nns.models.layers.disagreement_attention import inter_model
from nns.models.layers.disagreement_attention import intra_model
from nns.models.layers.disagreement_attention.constants import AttentionMergingType
from nns.segmentation.learning_algorithms import CoTraining, DACoTraining
from nns.segmentation.utils.postprocessing import ExpandPrediction


logzero.loglevel(settings.LOG_LEVEL)
# reproducibility = Reproducibility(
#     cuda=True, disable_cuda_benchmark=True, deterministic_algorithms=False, cublas_env_vars=True,
#     cuda_conv_determinism=True
# )


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
        metrics=settings.get_metrics(),
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
        metrics=settings.get_metrics(),
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
        metrics=settings.get_metrics(),
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
    #     metrics=settings.get_metrics(),
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
        metrics=settings.get_metrics(),
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
    #     metrics=settings.get_metrics(),
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
    # model6 = AEsModelMGR(
    #     # XAttentionUNet,  # XAttentionUNet,  # UNet_Att_DSV,  # UNet2D,  # UNet_Grid_Attention,  # AttentionUNet2, # UNet_3Plus,
    #     model=XAttentionAENet,
    #     # XAttentionAENet
    #     model_kwargs=dict(
    #         n_channels=3, n_classes=1, bilinear=False,
    #         batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
    #         data_dimensions=settings.DATA_DIMENSIONS, da_block_cls=intra_model.AttentionBlock,
    #         dsv=True, isolated_aes=False, true_aes=True, aes_loss=torch.nn.MSELoss(),  # torch.nn.L1Loss()
    #         out_ae_cls=MicroUpAE  # TinyUpAE, TinyAE, MicroUpAE, MicroAE
    #     ),
    #     # UNet4Plus
    #     # model_kwargs=dict(
    #     #     feature_scale=1, n_channels=3, n_classes=1, data_dimensions=settings.DATA_DIMENSIONS,
    #     #     is_batchnorm=True, batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
    #     #     dsv=False, multi_preds=True
    #     # ),
    #     # ModularUNet4Plus
    #     # model_kwargs=dict(
    #     #     feature_scale=1, n_channels=3, n_classes=1, data_dimensions=settings.DATA_DIMENSIONS,
    #     #     is_batchnorm=True, batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
    #     #     filters=[64, 128, 256, 512, 1024]
    #     # ),
    #     cuda=settings.CUDA,
    #     multigpus=settings.MULTIGPUS,
    #     patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
    #     epochs=30,  # 20
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
    #         # loss_functions.SpecificityLoss(with_logits=True),
    #     ],
    #     mask_threshold=0.5,
    #     metrics=settings.get_metrics(),
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
    class CTModelMGR(CT3DNIfTIMixin, ModelMGR):
        pass

    model7 = CTModelMGR(  # AEsModelMGR(
        model=XAttentionUNet,
        # UNet3D
        # model_kwargs=dict(feature_scale=1, n_channels=1, n_classes=1, is_batchnorm=True),
        # XAttentionUNet & XGridAttentionUNet & XAttentionUNet_ADSV
        model_kwargs=dict(
            n_channels=1, n_classes=1, bilinear=False, batchnorm_cls=get_batchnormxd_class(),
            init_type=UNet3InitMethod.KAIMING, data_dimensions=settings.DATA_DIMENSIONS,
            da_block_cls=intra_model.AttentionBlock,  # da_block_config={'xi': 1.},
            # da_block_config={'thresholds': (.25, .8), 'beta': -1},
            dsv=True,
        ),
        # XAttentionAENet
        # model_kwargs=dict(
        #     n_channels=1, n_classes=1, bilinear=False,
        #     batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
        #     data_dimensions=settings.DATA_DIMENSIONS, da_block_cls=intra_model.MixedEmbeddedDABlock,
        #     dsv=True, isolated_aes=False, true_aes=True, aes_loss=torch.nn.MSELoss(),  # torch.nn.L1Loss()
        #     out_ae_cls=MicroUpAE  # TinyUpAE, TinyAE, MicroUpAE, MicroAE
        # ),
        # Unet4Plus
        # model_kwargs=dict(
        #     feature_scale=1, n_channels=1, n_classes=1, data_dimensions=settings.DATA_DIMENSIONS,
        #     is_batchnorm=True, batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
        #     dsv=False, multi_preds=True
        # ),
        # ModularUNet4Plus
        # model_kwargs=dict(
        #     feature_scale=1, n_channels=1, n_classes=1, isolate=True, data_dimensions=settings.DATA_DIMENSIONS,
        #     is_batchnorm=True, batchnorm_cls=get_batchnormxd_class(), init_type=UNet3InitMethod.KAIMING,
        #     filters=[64, 128, 256, 512, 1024]
        # ),
        # UNet_Att_DSV
        # model_kwargs=dict(
        #     feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True,
        #     attention_block_cls=SingleAttentionBlock, data_dimensions=settings.DATA_DIMENSIONS
        # ),
        # UNet_Grid_Attention
        # model_kwargs=dict(
        #     feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True,
        #     data_dimensions=settings.DATA_DIMENSIONS
        # ),
        cuda=settings.CUDA,
        multigpus=settings.MULTIGPUS,
        patch_replication_callback=settings.PATCH_REPLICATION_CALLBACK,
        epochs=settings.EPOCHS,
        intrain_val=2,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6),
        sanity_checks=False,
        labels_data=LiTS17OnlyLiverLabels,  # LiTS17OnlyLesionLabels,  # CT82Labels,  # LiTS17OnlyLiverLabels
        data_dimensions=settings.DATA_DIMENSIONS,
        dataset=LiTS17CropDataset,  # CT82Dataset,  # LiTS17Dataset
        dataset_kwargs={
            'train_path': settings.LITS17_TRAIN_PATH,  # settings.CT82_TRAIN_PATH,  # settings.LITS17_TRAIN_PATH
            'val_path': settings.LITS17_VAL_PATH,   # settings.CT82_VAL_PATH,  # settings.LITS17_VAL_PATH
            'test_path': settings.LITS17_TEST_PATH,  # settings.CT82_TEST_PATH,  # settings.LITS17_TEST_PATH
            'cotraining': settings.COTRAINING,
            'cache': settings.DB_CACHE,
        },
        train_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS,
            'pin_memory': False,  # **reproducibility.dataloader_kwargs
        },
        testval_dataloader_kwargs={
            'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS,
            'pin_memory': False, 'drop_last': True,  # **reproducibility.dataloader_kwargs
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
        metrics=settings.get_metrics(),
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
        plot_dir=settings.PLOT_DIRECTORY,
        memory_print=dict(epochs=settings.EPOCHS//2),
    )
    model7()
    # model7.print_data_logger_summary()
    # model7.plot_and_save(None, 154)
    # id_ = '006'  # '004'
    # model7.predict(f'/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/test/cv_fold_5/CT_{id_}.nii.gz',
    #                patch_size=(32, 80, 80))
    # model7.plot_2D_ct_gt_preds(
    #     ct_path=f'/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/test/cv_fold_5/CT_{id_}.nii.gz',
    #     gt_path=f'/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/test/cv_fold_5/label_{id_}.nii.gz',
    #     pred_path=f'pred_CT_{id_}.nii.gz',
    #     only_slices_with_masks=True, save_to_disk=True, dpi=300, no_axis=True, tight_layout=False,
    #     max_slices=62
    # )
    # summary(model6.module, (4, 3, *settings.CROP_IMG_SHAPE), depth=1, verbose=1)
    # m = UNet3D(feature_scale=1, n_classes=1, n_channels=1, is_batchnorm=True)
    # settings.LITS17_CROP_SHAPE
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
    #     metrics=settings.get_metrics(),
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
    # mgr.split_processed_dataset(.20, .20, shuffle=False)  # to easily apply 5-fold CV later

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

    #     # NIfTI.save_numpy_as_nifti(data['image'].detach().cpu().squeeze().permute(
    #     #     1, 2, 0).numpy(), f'{db_name}_img_patch.nii.gz')
    #     # NIfTI.save_numpy_as_nifti(data['mask'].detach().cpu().squeeze().permute(
    #     #     1, 2, 0).numpy(), f'{db_name}_mask_patch.nii.gz')

    #     print(data['image'].shape, data['mask'].shape)
    #     print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])

    #     print(data['image'].min(), data['image'].max())
    #     print(data['mask'].min(), data['mask'].max())

    #     if len(data['image'].shape) == 4:
    #         img_id = np.random.randint(0, data['image'].shape[-3])
    #         fig, axis = plt.subplots(1, 2)
    #         axis[0].imshow(
    #             equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #             cmap='gray'
    #         )
    #         axis[1].imshow(
    #             data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()
    #     else:
    #         fig, axis = plt.subplots(2, 4)
    #         for idx, d in zip([*range(4)], dataset):
    #             img_id = np.random.randint(0, d['image'].shape[-3])
    #             axis[0, idx].imshow(
    #                 equalize_adapthist(d['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #                 cmap='gray'
    #             )
    #             axis[1, idx].imshow(
    #                 d['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
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
    ###########################################################################
    #                                  LiTS17                                 #
    ###########################################################################
    # labels files: 131, CT files: 131
    #                           value
    # ------------------------  ---------------------------------------------------------
    # Files without label 1     []
    # Files without label 2     [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
    # Total CT files            131
    # Total segmentation files  131
    #
    #                         min    max
    # -------------------  ------  -----
    # Image value          -10522  27572
    # Slices with label 1      28    299
    # Slices with label 2       0    245
    # Height                  512    512
    # Width                   512    512
    # Depth                    74    987

    # mgr = LiTS17MGR('/media/giussepi/TOSHIBA EXT/LITS/train',
    #                 saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro',
    #                 target_size=(368, 368, -1), only_liver=False, only_lesion=True)
    # mgr.get_insights(verbose=True)
    # print(mgr.get_lowest_highest_bounds())
    # mgr()
    # mgr.perform_visual_verification(68, scans=[40, 64], clahe=True)  # ppl 68 -> scans 64
    # after manually removing files without the desired label and less scans than 32
    # (000, 001, 054 had 29 scans) we ended up with 230 FILES @ LiTS17 only lesion and
    # 256 files @ LiTS17 only liver
    # mgr.split_processed_dataset(.20, .20, shuffle=True)

    # min_ = float('inf')
    # max_ = float('-inf')
    # min_scans = float('inf')
    # max_scans = float('-inf')
    # for f in tqdm(glob.glob('/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro/**/label_*.nii.gz', recursive=True)):
    #     data = NIfTI(f).ndarray
    #     num_scans_with_labels = data.sum(axis=0).sum(axis=0).astype(bool).sum()
    #     min_scans = min(min_scans, data.shape[-1])
    #     max_scans = max(max_scans, data.shape[-1])
    #     min_ = min(min_, num_scans_with_labels)
    #     max_ = max(max_, num_scans_with_labels)
    #     assert len(np.unique(data)) == 2
    #     assert 1 in np.unique(data)
    #     # print(np.unique(NIfTI(f).ndarray))
    # print(min_, max_, min_scans, max_scans)
    # @LiTS17Lesion-Pro the min, max number of scans with dataper label are  3 and 245 !!!!
    #                       min, max scans are 29, 299

    # analyzing number of scans on generated lists17 with only liver label ####
    # counter = defaultdict(lambda: 0)
    # for f in tqdm(glob.glob('/media/giussepi/TOSHIBA EXT/LiTS17Liver-Pro/**/label_*.nii.gz', recursive=True)):
    #     scans = NIfTI(f).shape[-1]
    #     counter[scans] += 1
    #     if scans == 29:
    #         print(f)
    # a = [*counter.keys()]
    # a.sort()
    # print(a)
    # print(counter[29])
    # @LiTS17Liver-Pro the labels are [29, 32, 26, ..., 299
    # and we only have 3 cases with 29 scans so we can get rid of them to
    # use the same crop size as CT-32
    # these cases are the 000, 001, 054

    # getting subdatasets and plotting some crops #############################
    # train, val, test = LiTS17Dataset.get_subdatasets(
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro/train',
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro/val',
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro/test'
    # )
    # for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    #     print(f'{db_name}: {len(dataset)}')
    #     data = dataset[0]
    #     print(data['image'].shape, data['mask'].shape)
    #     print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
    #     print(data['image'].min(), data['image'].max())
    #     print(data['mask'].min(), data['mask'].max())

    #     if len(data['image'].shape) == 4:
    #         img_id = np.random.randint(0, data['image'].shape[-3])
    #         fig, axis = plt.subplots(1, 2)
    #         axis[0].imshow(
    #             equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #             cmap='gray'
    #         )
    #         axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #         plt.show()
    #     else:
    #         num_crops = dataset[0]['image'].shape[0]
    #         imgs_per_row = 4
    #         for ii in range(0, len(dataset), imgs_per_row):
    #             fig, axis = plt.subplots(2, imgs_per_row*num_crops)
    #             # for idx, d in zip([*range(imgs_per_row)], dataset):
    #             for idx in range(imgs_per_row):
    #                 d = dataset[idx+ii]
    #                 for cidx in range(num_crops):
    #                     img_id = np.random.randint(0, d['image'].shape[-3])
    #                     axis[0, idx*num_crops+cidx].imshow(
    #                         equalize_adapthist(d['image'][cidx].detach().numpy()
    #                                            ).squeeze().transpose(1, 2, 0)[..., img_id],
    #                         cmap='gray'
    #                     )
    #                     axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
    #                     axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
    #                     ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                     axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

    #             fig.suptitle('CTs and Masks')
    #             plt.show()

    # generating crops dataset ################################################
    # LiTS17CropMGR(
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro',
    #     patch_size=tuple([*settings.LITS17_CROP_SHAPE[1:], settings.LITS17_CROP_SHAPE[0]]),
    #     patch_overlapping=(.25, .25, .25), only_crops_with_masks=True, min_mask_area=25e-4,
    #     min_crop_mean=0.41, crops_per_label=20, adjust_depth=False,
    #     saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-20PositiveCrops'
    # )()
    # return 1
    # Total crops: 3853
    # Label 2 crops: 0
    # Label 1 crops: 1553
    # Label 0 crops: 2300
    # train 2291, val 762, test 800
    # Total crops: 1553
    # Label 2 crops: 0
    # Label 1 crops: 1553
    # all crops with masks [I 220928 18:17:29 lits17cropmgr:133] Total crops: 2683
    # min_mask_area 25e-4: 15, 25, 73 has 1 label file  # slice area 80x80x25e-4 = 16
    # min_mask_area 1e-15: 73 -> 1

    # creating crop lesion dataset 64x160x160 #################################
    # mgr = LiTS17MGR('/media/giussepi/TOSHIBA EXT/LITS/train',
    # saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion368x368x-2-Pro',
    # target_size=(368, 368, -2), only_liver=False, only_lesion=True)
    # mgr.get_insights(verbose=True)
    # print(mgr.get_lowest_highest_bounds())
    # mgr()
    # mgr.verify_generated_db_target_size()
    # mgr.perform_visual_verification(68, scans=[40, 64], clahe=True)  # ppl 68 -> scans 64
    # after manually removing Files without label 2
    # [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
    # mgr.split_processed_dataset(.20, .20, shuffle=True)
    # we aim to work with crops masks with an minimum area of 16 so min_mask_area
    # for the following heightxheight crops are:
    # 80x80x25e-4 = 16
    # 160x160x625e-6 = 16
    # LiTS17CropMGR(
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion368x368x-2-Pro',
    #     patch_size=tuple([*settings.LITS17_CROP_SHAPE[1:], settings.LITS17_CROP_SHAPE[0]]),
    #     patch_overlapping=(.75, .75, .75), only_crops_with_masks=True, min_mask_area=625e-6,
    #     foregroundmask_threshold=.59, min_crop_mean=.63, crops_per_label=16, adjust_depth=True,
    #     centre_masks=True,
    #     saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-16PositiveCrops32x160x160'
    # )()
    # return 1
    # crops_per_label=20
    # Total crops: 1212
    # Label 2 crops: 0
    # Label 1 crops: 1212
    # Label 0 crops: 0
    # crops_per_label=8
    # Total crops created: 754
    # Label 2 crops: 0
    # Label 1 crops: 754
    # Label 0 crops: 0
    # crops per lael = 4
    # Total crops created: 472
    # Label 2 crops: 0
    # Label 1 crops: 472
    # Label 0 crops: 0
    # crops per label = 16
    # Total crops created: 1888
    # Label 2 crops: 0
    # Label 1 crops: 1888
    # Label 0 crops: 0
    # getting subdatasets and plotting some crops #############################
    # train, val, test = LiTS17CropDataset.get_subdatasets(
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-16PositiveCrops32x160x160/train',
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-16PositiveCrops32x160x160/val',
    #     '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-16PositiveCrops32x160x160/test'
    # )
    # for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    #     print(f'{db_name}: {len(dataset)}')
    #     # for _ in tqdm(dataset):
    #     #     pass
    #     # data = dataset[0]

    #     for data_idx in range(len(dataset)):
    #         data = dataset[data_idx]
    #         # print(data['image'].shape, data['mask'].shape)
    #         # print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
    #         # print(data['image'].min(), data['image'].max())
    #         # print(data['mask'].min(), data['mask'].max())
    #         if len(data['image'].shape) == 4:
    #             img_ids = [np.random.randint(0, data['image'].shape[-3])]

    #             # uncomment these lines to only plot crops with masks
    #             # if 1 not in data['mask'].unique():
    #             #     continue
    #             # else:
    #             #     # selecting an idx containing part of the mask
    #             #     img_ids = data['mask'].squeeze().sum(axis=-1).sum(axis=-1).nonzero().squeeze()

    #             foreground_mask = ForegroundMask(threshold=.59, invert=True)(data['image'])
    #             std, mean = torch.std_mean(data['image'], unbiased=False)
    #             fstd, fmean = torch.std_mean(foreground_mask, unbiased=False)

    #             # once you have chosen a good mean, uncomment the following
    #             # lines and replace .63 with your chosen mean to verify that
    #             # only good crops are displayed.
    #             # if fmean < .63:
    #             #     continue

    #             print(f"SUM: {data['image'].sum()}")
    #             print(f"STD MEAN: {std} {mean}")
    #             print(f"SUM: {foreground_mask.sum()}")
    #             print(f"foreground mask STD MEAN: {fstd} {fmean}")

    #             for img_id in img_ids:
    #                 fig, axis = plt.subplots(1, 3)
    #                 axis[0].imshow(
    #                     equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
    #                     cmap='gray'
    #                 )
    #                 axis[0].set_title('Img')
    #                 axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                 axis[1].set_title('mask')
    #                 axis[2].imshow(foreground_mask.detach().numpy().squeeze()
    #                                .transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                 axis[2].set_title('foreground_mask')
    #                 plt.show()
    #                 plt.clf()
    #                 plt.close()
    #         else:
    #             num_crops = dataset[0]['image'].shape[0]
    #             imgs_per_row = 4
    #             for ii in range(0, len(dataset), imgs_per_row):
    #                 fig, axis = plt.subplots(2, imgs_per_row*num_crops)
    #                 # for idx, d in zip([*range(imgs_per_row)], dataset):
    #                 for idx in range(imgs_per_row):
    #                     d = dataset[idx+ii]
    #                     for cidx in range(num_crops):
    #                         img_id = np.random.randint(0, d['image'].shape[-3])
    #                         axis[0, idx*num_crops+cidx].imshow(
    #                             equalize_adapthist(d['image'][cidx].detach().numpy()
    #                                                ).squeeze().transpose(1, 2, 0)[..., img_id],
    #                             cmap='gray'
    #                         )
    #                         axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
    #                         axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
    #                         ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
    #                         axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

    #                 fig.suptitle('CTs and Masks')
    #                 plt.show()

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

    ###########################################################################
    #                                  CoNSeP                                 #
    ###########################################################################
    # https://warwick.ac.uk/fac/cross_fac/tia/data/
    # The colorectal nuclear segmentation and phenotypes(CoNSeP) dataset
    # consists of 41 H & E stained image tiles, each of size 1, 000×1, 000 pixels
    # at 40× objective magnification. The images were extracted from 16
    # colorectal adenocarcinoma(CRA) WSIs, each belonging to an individual
    # patient, and scanned with an Omnyx VL120 scanner within the department of
    # pathology at University Hospitals Coventry and Warwickshire, UK.

    # new dataset of Haematoxylin & Eosin stained colorectal adenocarcinoma
    # image tiles, containing 24,319 exhaustively annotated nuclei with
    # associated class labels.

    # consep 26 train 14 test, all 41 images extracted from 16 WSIs
    ###############################################################################


if __name__ == '__main__':
    main()
