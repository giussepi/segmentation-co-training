# -*- coding: utf-8 -*-
""" nns/mixins/managers/standard_managers """

from nns.mixins.managers.base import BaseModelMGR


__all__ = ['ModelMGRMixin']


class ModelMGRMixin(BaseModelMGR):
    """
    General segmentation model manager

    Usage:
        class MyModelMGR(ModelMGRMixin):
           ...

        model = MyModelMGR(
            model=UNet,
            model_kwargs=dict(n_channels=3, n_classes=10, bilinear=True),
            cuda=True,
            multigpus=True,
            patch_replication_callback=False,
            epochs=settings.EPOCHS,
            intrain_val=2,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            sanity_checks=True,
            labels_data=MyLabelClass,
            data_dimensions=2,
            ###################################################################
            #                         SubDatasetsMixin                         #
            ###################################################################
            dataset=CRLMMultiClassDataset,
            dataset_kwargs={
                'train_path': settings.CRLM_TRAIN_PATH,
                'val_path': settings.CRLM_VAL_PATH,
                'test_path': settings.CRLM_TEST_PATH,
            },
            train_dataloader_kwargs={
                'batch_size': 2, 'shuffle': True, 'num_workers': 12, 'pin_memory': False
            },
            testval_dataloader_kwargs={
                'batch_size': 2, 'shuffle': False, 'num_workers': 12, 'pin_memory': False, 'drop_last': True
            },
            ###################################################################
            lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
            lr_scheduler_kwargs={'mode': 'max', 'patience': 2},  # {'step_size': 10, 'gamma': 0.1},
            lr_scheduler_track=LrShedulerTrack.NO_ARGS,
            criterions=[
                # nn.BCEWithLogitsLoss()
                # MSSSIMLoss(window_size=11, size_average=True, channel=3),
                # nn.BCELoss(size_average=True),
                # IOULoss(),
                # nn.CrossEntropyLoss()
                # bce_dice_loss  # 1866.6306
                # bce_dice_loss_ # 1890.8262
                DC_RNPV_LOSS(dice_threshold=0.25, always_conditioned=True)
            ],
            mask_threshold=0.5,
            metrics=[
                MetricItem(torchmetrics.DiceCoefficient(), main=True),
                MetricItem(torchmetrics.Specificity(), main=True),
                MetricItem(torchmetrics.Recall())
            ],
            metric_mode=MetricEvaluatorMode.MAX,
            earlystopping_kwargs=dict(min_delta=1e-3, patience=np.inf, metric=True),
            checkpoint_interval=1,
            train_eval_chkpt=True,
            last_checkpoint=True,
            ini_checkpoint='',
            dir_checkpoints=settings.DIR_CHECKPOINTS,
            tensorboard=True,
            # TODO: there a bug that appeared once when plotting to disk after a long training
            # anyway I can always plot from the checkpoints :)
            plot_to_disk=False,
            plot_dir=settings.PLOT_DIRECTORY,
            memory_print=dict(epochs=settings.EPOCHS//2),
        )()
    """
