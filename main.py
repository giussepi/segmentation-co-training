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
from nns.backbones import resnet101, resnet152, xception
from nns.callbacks.metrics.constants import MetricEvaluatorMode
from nns.managers import ModelMGR, DAModelMGR
from nns.mixins.constants import LrShedulerTrack
from nns.models import Deeplabv3plus, UNet_3Plus_DA, UNet_3Plus_DA_Train, UNet_3Plus_DA2, \
    UNet_3Plus_DA2_Train, UNet_3Plus_DA2Ext, UNet_3Plus_DA2Ext_Train
from nns.models.layers.disagreement_attention import ThresholdedDisagreementAttentionBlock, \
    MergedDisagreementAttentionBlock, PureDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.constants import AttentionMergingType
from nns.segmentation.learning_algorithms import CoTraining, DACoTraining
from nns.segmentation.utils.postprocessing import ExpandPrediction
from nns.utils.sync_batchnorm import get_batchnorm2d_class


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
                          batchnorm_cls=get_batchnorm2d_class()),
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
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'unet3_plus_1'),
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
                          batchnorm_cls=get_batchnorm2d_class()),
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
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'unet3_plus_2'),
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
            batchnorm=get_batchnorm2d_class(), backbone=xception, backbone_pretrained=True,
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
            settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'deeplabv3plus_xception'),
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
    #     dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85'),
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
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'chkpt_4.pth.tar'))

    #     cot.plot_and_save(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'chkpt_4.pth.tar'),
    #         save=True, show=False, dpi=300.
    #     )

    #     cot.print_data_logger_details(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'chkpt_4.pth.tar'))
    # except:
    #     pass

    # DA experiments ##########################################################
    # TODO: update doctrings from DAModelMGRMixin
    # model4 = dict(
    model4 = DAModelMGR(
        model_cls=UNet_3Plus_DA_Train,
        model_kwargs=dict(
            model1_cls=UNet_3Plus_DA,
            kwargs1=dict(da_threshold=np.NINF, da_block_cls=ThresholdedDisagreementAttentionBlock,
                         da_block_config=dict(thresholds=(.25, .8), beta=0.),
                         # da_merging_type=AttentionMergingType.MAX,
                         n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.XAVIER,
                         batchnorm_cls=get_batchnorm2d_class()),
            model2_cls=UNet_3Plus_DA,
            kwargs2=dict(da_threshold=np.NINF, da_block_cls=ThresholdedDisagreementAttentionBlock,
                         da_block_config=dict(thresholds=(.25, .8), beta=0.),
                         # da_merging_type=AttentionMergingType.MAX,
                         n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.KAIMING,
                         batchnorm_cls=get_batchnorm2d_class()),
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
            # loss_functions.BceDiceLoss(with_logits=True),
            BceDiceLoss(),
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
        dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'unet3_plus_DA'),
        tensorboard=False,
        # TODO: there a bug that appeared once when plotting to disk after a long training
        # anyway I can always plot from the checkpoints :)
        plot_to_disk=False,
        plot_dir=settings.PLOT_DIRECTORY
    )
    # summary(model4.module, depth=10, verbose=1)
    model4()
    # model4.predict('1.ann.tiff', Image.open, patch_size=256, patch_overlapping=2, superimpose=False, size=None)
    # model4.print_data_logger_summary()
    # _, data_logger = model4.load_checkpoint([
    #     model4.optimizer1(model4.module.model1.parameters(), **model4.optimizer1_kwargs),
    #     model4.optimizer2(model4.module.model2.parameters(), **model4.optimizer2_kwargs),
    # ])

    # __import__("pdb").set_trace()

    # model4.plot_and_save(152)

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
    #     dir_checkpoints=os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85'),
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
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'chkpt_4.pth.tar'))

    #     cot.plot_and_save(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'chkpt_4.pth.tar'),
    #         save=True, show=False, dpi=300.
    #     )

    #     cot.print_data_logger_details(
    #         os.path.join(settings.DIR_CHECKPOINTS, 'consep', 'cotraining', 'exp85', 'chkpt_4.pth.tar'))
    # except Exception:
    #     pass


if __name__ == '__main__':
    main()
