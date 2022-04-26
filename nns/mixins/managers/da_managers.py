# -*- coding: utf-8 -*-
""" nns/mixins/managers/da_managers """

import os
import sys
from copy import deepcopy
from itertools import chain
from statistics import mean
from typing import Callable, Optional
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import openslide as ops
import torch
from gtorch_utils.nns.managers.callbacks import Checkpoint, EarlyStopping
from gutils.decorators import timing
from gutils.folders import clean_create_folder
from gutils.images.processing import get_slices_coords
from gutils.images.postprocessing import RemoveBG
from PIL import Image
from logzero import logger
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nns.callbacks.metrics import MetricEvaluator
from nns.callbacks.metrics.constants import MetricEvaluatorMode
from nns.callbacks.plotters.masks import MaskPlotter
from nns.mixins.constants import LrShedulerTrack
from nns.mixins.checkpoints import DACheckPointMixin
from nns.mixins.data_loggers import DADataLoggerMixin
from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR
from nns.mixins.subdatasets import SubDatasetsMixin
from nns.models.da_model import BaseDATrain
from nns.utils.sync_batchnorm import patch_replication_callback


__all__ = ['DAModelMGRMixin']


class DAModelMGRMixin(DACheckPointMixin, DADataLoggerMixin, SubDatasetsMixin):
    """
    Segmentation model manager mixin for disagreement attention

    Usage:
        class MyModelMGR(ModelMGRMixin):
           ...

        model = MyModelMGR(
            model_cls=UNet_3Plus_DA_Train,
            model_kwargs=dict(
                model1_cls=UNet_3Plus_DA,
                kwargs1=dict(da_threshold=np.NINF, da_block_config=dict(thresholds=(.25, .8), beta=-1.),
                             n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.XAVIER,
                             batchnorm_cls=get_batchnorm2d_class()),
                model2_cls=UNet_3Plus_DA,
                kwargs2=dict(da_threshold=np.NINF, da_block_config=dict(thresholds=(.25, .8), beta=-1.),
                             n_channels=3, n_classes=1, is_deconv=False, init_type=UNet3InitMethod.XAVIER,
                             batchnorm_cls=get_batchnorm2d_class()),
            ),
            cuda=True,
            multigpus=True,
            patch_replication_callback=False,
            epochs=10,
            intrain_val=2,
            optimizer1=torch.optim.Adam,
            optimizer1_kwargs=dict(lr=1e-4),
            optimizer2=torch.optim.Adam,
            optimizer2_kwargs=dict(lr=1e-4),
            labels_data=MyLabelClass,
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
            lr_scheduler1=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
            # TODO: the mode can change based on the quantity monitored
            # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
            lr_scheduler1_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
            lr_scheduler1_track=LrShedulerTrack.LOSS,
            lr_scheduler2=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
            lr_scheduler2_kwargs={'mode': 'min', 'patience': 4},  # {'step_size': 10, 'gamma': 0.1},
            lr_scheduler2_track=LrShedulerTrack.LOSS,
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
            plot_dir=settings.PLOT_DIRECTORY
        )()
    """

    def __init__(self, **kwargs):
        """
        Kwargs:
            model_cls        <BaseDATrain>: Class descendant of BaseDATrain with a customized forward pass
                                            to appropriately train model1 and model2 using disagreement
                                            attention
            model_kwargs       <nn.Module>: Dictionary containing the data create an instance of
                                            model_cls
            model1_kwargs           <dict>: Dictionary holding the initial arguments for model 1
            model2_kwargs           <dict>: Dictionary holding the initial arguments for model 2
            cuda                    <bool>: whether or not use cuda
            multigpus               <bool>: Set it to True to use several GPUS. It requires to set
                                            cuda to True. Default False
            patch_replication_callback <bool>: Whether or not to apply patch_replication_callback. It must
                                            be used only if SynchronizedBatchNorm2d has been used.
                                            Furthermore, it requires that
                                            cuda = multigpus = patch_replication_callback = True and
                                            there are multiple GPUs available. Default = False
            epochs                   <int>: number of epochs
            intrain_val              <int>: Times to interrupt the iteration over the training dataset
                                            to collect statistics, perform validation and update
                                            the learning rate. Default 10
            optimizer1         <Optimizer>: optimizer class from torch.optim for model 1
            optimizer2         <Optimizer>: optimizer class from torch.optim for model 2
            optimizer1_kwargs       <dict>: keyword arguments for optimizer 1
            optimizer2_kwargs       <dict>: keyword arguments for optimizer 2
            labels_data           <object>: class containing all the details of the classes/labels. See
                                            nns.callbacks.plotters.masks.MaskPlotter definition
            ###################################################################
            #                         SubDatasetsMixin                        #
            ###################################################################
            dataset      <DatasetTemplate>: Custom dataset class descendant of
                                            gtorch_utils.datasets.segmentation.DatasetTemplate.
                                            See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
            dataset_kwargs          <dict>: keyword arguments for the dataset
            train_dataloader_kwargs <dict>: Keyword arguments for the train DataLoader
            testval_dataloader_kwargs <dict>: Keyword arguments for the test and validation DataLoaders
            ###################################################################
            lr_scheduler1         <object>: learning rate scheduler class from torch.optim.lr_scheduler
                                            for model 1
            lr_scheduler1_kwargs    <dict>: keyword arguments for lr_scheduler1. Some lr_schedulers
                                            like ReduceLROnPlateau have the option 'mode' to speficy if
                                            the value provided when calling scheduler.step(val_metric)
                                            must be maximized or minimized. If your metric is better when
                                            it is close to 1 then set it to max, otherwise to min. On the
                                            other hand, if you use the loss with the lr_scheduler then you
                                            should set 'mode' to min. If patience is provided, it will be
                                            multiplied by intrain_val to reduce the learning rate after
                                            'patience' epochs
            lr_scheduler1_track     <bool>: Defines if the lr_scheduler.step, for optimizer1, should be
                                            called with val_loss, val_metric o no arguments. See
                                            nns.mixins.constants.LrShedulerTrack
                                            Default LrShedulerTrack.NO_ARGS
            lr_scheduler2         <object>: learning rate scheduler class from torch.optim.lr_scheduler
                                            for model 2
            lr_scheduler2_kwargs    <dict>: keyword arguments for lr_scheduler2. Some lr_schedulers
                                            like ReduceLROnPlateau have the option 'mode' to speficy if
                                            the value provided when calling scheduler.step(val_metric)
                                            must be maximized or minimized. If your metric is better when
                                            it is close to 1 then set it to max, otherwise to min. On the
                                            other hand, if you use the loss with the lr_scheduler then you
                                            should set 'mode' to min. If patience is provided, it will be
                                            multiplied by intrain_val to reduce the learning rate after
                                            'patience' epochs
            lr_scheduler2_track     <bool>: Defines if the lr_scheduler.step, for optimizer2, should be
                                            called with val_loss,
                                            val_metric o no arguments. See
                                            nns.mixins.constants.LrShedulerTrack
                                            Default LrShedulerTrack.NO_ARGS
            criterions              <list>: List of one of more losses
            mask_threshold         <float>: mask threshold. Default 0.5
            metrics                 <list>: List of MetricItems to be used by the manager
                                            Default [MetricItem(DiceCoefficient(), main=True),]
            metric_mode              <int>: Evaluation mode of the metric.
                                            See nns.callbacks.metrics.constants.MetricEvaluatorMode
                                            Default MetricEvaluatorMode.MAX
            earlystopping_kwargs    <dict>: Early stopping parameters. When metric = True, it is
                                            applied to the metric values; otherwise, it is applied to the
                                            loss values. If patience is provided, it will be multiplied
                                            by intrain_val to stop everything after 'patience' epochs.
                                            To disable it just set patience = np.inf
                                            See gtorch_utils.nns.managers.callbacks.EarlyStopping class
                                            definition
                                            Default dict(min_delta=1e-3, patience=8, metric=True)
            checkpoint_interval      <int>: interval of epochs before saving a checkpoint.
                                            If <= 0 then checkpoints are not saved.
                                            Default 1
            train_eval_chkpt        <bool>: If True, a checkpoint will be saved right after each evaluation
                                            executed while processing the training subdataset
                                            (e.gl chkpt_1.1.pth.tar) Default False
            last_checkpoint         <bool>: If True, the last checkpoint will be saved separately. This is
                                            useful when checkpoint_interval is set to zero to save disk
                                            space and you want to have the last checkpoint to get all the
                                            statistics from the whole training process. Default False
            ini_checkpoint           <str>: path to checkpoint to load. So the training can continue.
                                            It must be inside the the dir_checkpoints directory.
                                            Default ''
            dir_checkpoint           <str>: path to the directory where checkpoints will be saved
            tensorboard             <bool>: whether or not plot training data into tensorboard. Default True
            plot_to_disk            <bool>: Whether or not plot data training data and save it as images.
                                            Default True
            plot_dir                 <str>: Directory where the training plots will be saved. Default 'plots'
        """
        self.model_cls = kwargs.get('model_cls')
        self.model_kwargs = kwargs.get('model_kwargs')
        self.cuda = kwargs.get('cuda', True)
        self.multigpus = kwargs.get('multigpus', False)
        self.patch_replication_callback = kwargs.get('patch_replication_callback', False)
        self.epochs = kwargs.get('epochs', 5)
        self.intrain_val = kwargs.get('intrain_val', 10)
        self.optimizer1 = kwargs.get('optimizer1', torch.optim.RMSprop)
        self.optimizer1_kwargs = kwargs.get(
            'optimizer1_kwargs', dict(lr=1e-4, weight_decay=1e-8, momentum=.9))
        self.optimizer2 = kwargs.get('optimizer2', torch.optim.RMSprop)
        self.optimizer2_kwargs = kwargs.get(
            'optimizer2_kwargs', dict(lr=1e-4, weight_decay=1e-8, momentum=.9))
        self.labels_data = kwargs['labels_data']
        self.dataset = kwargs['dataset']

        self.lr_scheduler1 = kwargs.get('lr_scheduler1', None)
        self.lr_scheduler1_kwargs = kwargs.get('lr_scheduler1_kwargs', {})
        self.lr_scheduler1_track = kwargs.get('lr_scheduler1_track', LrShedulerTrack.NO_ARGS)
        self.lr_scheduler2 = kwargs.get('lr_scheduler2', None)
        self.lr_scheduler2_kwargs = kwargs.get('lr_scheduler2_kwargs', {})
        self.lr_scheduler2_track = kwargs.get('lr_scheduler2_track', LrShedulerTrack.NO_ARGS)
        self.criterions = kwargs.get('criterions', None)  # same criterions applied to both NNs

        if not self.criterions:
            if self.module.n_classes > 1:
                self.criterions = [nn.CrossEntropyLoss()]
            else:
                self.criterions = [nn.BCEWithLogitsLoss()]

        self.mask_threshold = kwargs.get('mask_threshold', 0.5)
        self.metric_mode = kwargs.get('metric_mode', MetricEvaluatorMode.MAX)
        self.earlystopping_kwargs = kwargs.get(
            'earlystopping_kwargs', dict(min_delta=1e-3, patience=8, metric=True))
        self.earlystopping_to_metric = self.earlystopping_kwargs.pop('metric')
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 1)
        self.train_eval_chkpt = kwargs.get('train_eval_chkpt', False)
        self.last_checkpoint = kwargs.get('last_checkpoint', False)
        self.ini_checkpoint = kwargs.get('ini_checkpoint', '')
        self.dir_checkpoints = kwargs.get('dir_checkpoints', 'checkpoints')
        self.tensorboard = kwargs.get('tensorboard', True)
        self.plot_to_disk = kwargs.get('plot_to_disk', True)
        self.plot_dir = kwargs.get('plot_dir', 'plots')

        assert issubclass(self.model_cls, BaseDATrain), type(self.model_cls)
        assert isinstance(self.model_kwargs, dict), type(self.model_kwargs)
        assert isinstance(self.cuda, bool), type(self.cuda)
        assert isinstance(self.multigpus, bool), type(self.multigpus)
        assert isinstance(self.patch_replication_callback, bool), type(self.patch_replication_callback)
        assert isinstance(self.epochs, int), type(self.epochs)
        assert isinstance(self.intrain_val, int), type(self.intrain_val)
        assert self.intrain_val >= 1, self.intrain_val
        assert issubclass(self.optimizer1, Optimizer), 'optimizer1 must be a subclass of Optimizer'
        assert isinstance(self.optimizer1_kwargs, dict), type(self.optimizer1_kwargs)
        assert issubclass(self.optimizer2, Optimizer), 'optimizer2 must be a subclass of Optimizer'
        assert isinstance(self.optimizer2_kwargs, dict), type(self.optimizer2_kwargs)
        assert issubclass(self.lr_scheduler1, object), type(self.lr_scheduler1)
        assert isinstance(self.lr_scheduler1_kwargs, dict), type(self.lr_scheduler1_kwargs)
        LrShedulerTrack.validate(self.lr_scheduler1_track)
        assert issubclass(self.lr_scheduler2, object), type(self.lr_scheduler2)
        assert isinstance(self.lr_scheduler2_kwargs, dict), type(self.lr_scheduler2_kwargs)
        LrShedulerTrack.validate(self.lr_scheduler2_track)
        assert isinstance(self.criterions, list), type(self.criterions)
        assert isinstance(self.mask_threshold, float), type(self.mask_threshold)
        MetricEvaluatorMode.validate(self.metric_mode)
        assert isinstance(self.earlystopping_kwargs, dict), type(self.earlystopping_kwargs)
        assert isinstance(self.checkpoint_interval, int), type(self.checkpoint_interval)
        assert isinstance(self.train_eval_chkpt, bool), type(self.train_eval_chkpt)
        assert isinstance(self.last_checkpoint, bool), type(self.last_checkpoint)
        assert isinstance(self.ini_checkpoint, str), type(self.ini_checkpoint)

        if self.ini_checkpoint:
            assert os.path.isfile(os.path.join(self.dir_checkpoints, self.ini_checkpoint)), \
                self.ini_checkpoint

        assert isinstance(self.dir_checkpoints, str), type(self.dir_checkpoints)

        if not os.path.isdir(self.dir_checkpoints):
            os.makedirs(self.dir_checkpoints)

        assert isinstance(self.tensorboard, bool), type(self.tensorboard)
        assert isinstance(self.plot_to_disk, bool), type(self.plot_to_disk)
        assert isinstance(self.plot_dir, str), type(self.plot_dir)

        # updating the earlystopping patience according to the number of 'intrain_val'
        # to stop, if necessary, the whole process after 'patience' epochs
        self.earlystopping_kwargs['patience'] *= self.intrain_val

        # updating the lr_scheduler patience according to the number of 'intrain_val'
        # to reduce the learning rate after 'patience' epochs
        if 'patience' in self.lr_scheduler1_kwargs:
            self.lr_scheduler1_kwargs['patience'] *= self.intrain_val
        if 'patience' in self.lr_scheduler2_kwargs:
            self.lr_scheduler2_kwargs['patience'] *= self.intrain_val

        if self.plot_to_disk and self.plot_dir and not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.model = self.model_cls(**self.model_kwargs)

        if self.cuda:
            if self.multigpus:
                self.model = nn.DataParallel(self.model)

            if isinstance(self.model, nn.DataParallel):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if torch.cuda.is_available() and torch.cuda.device_count() > 1 and \
                   self.patch_replication_callback:
                    # TODO: I think this line is properly linking the patch replication
                    # with the dataparallel. Double-check it with some tests
                    patch_replication_callback(self.model)
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            if self.device == "cpu":
                logger.warning("CUDA is not available. Using CPU")
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self._SubDatasetsMixin__init(**kwargs)
        self._TorchMetricsBaseMixin__init(**kwargs)

    # same
    @staticmethod
    def calculate_loss(criterion_list, pred_masks, true_masks):
        """
        Calculates the summation of the losses calculated using the criterion list

        Args:
            criterion          <list>:
            masks_pred <torch.Tensor>: predicted masks
            true_masks <torch.Tensor>: ground truth masks

        Returns:
            loss <torch.Tensor>
        """
        assert isinstance(criterion_list, list), type(criterion_list)
        assert isinstance(pred_masks, torch.Tensor), type(pred_masks)
        assert isinstance(true_masks, torch.Tensor), type(true_masks)

        losses = torch.stack([criterion(pred_masks, true_masks) for criterion in criterion_list])

        return torch.sum(losses)

    # same
    def __call__(self):
        """ functor call """
        self.fit()
        self.test()

    # same
    @property
    def module(self):
        """
        Returns the model

        Returns:
            model <nn.Module>
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module

        return self.model

    # same
    @staticmethod
    def reshape_data(imgs, labels, true_masks=None, filepaths=None):
        """
        Reshapes the tensors to be properly used

        Args:
            imgs       <Tensor>: Tensor containing the images
            labels     <Tensor>: Tensor containing the labels
            true_masks <Tensor>: Tensor containing the ground truth masks
            filepaths  <Tensor>: Tensor containing the filepaths

        Returns:
            imgs, labels, true_masks, filepaths
        """
        assert torch.is_tensor(imgs)
        assert torch.is_tensor(labels)

        if true_masks is not None:
            assert torch.is_tensor(true_masks)
            true_masks = true_masks.reshape((-1, *true_masks.shape[2:]))

        if filepaths is not None:
            assert torch.is_tensor(filepaths)
            filepaths = filepaths.squeeze()

        imgs = imgs.reshape((-1, *imgs.shape[2:]))
        labels = labels.squeeze()

        return imgs, labels, true_masks, filepaths

    # same
    @staticmethod
    def basic_preprocess(img):
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

    def get_validation_data(self, batch: dict):
        """
        Returns the data to be used for the validation or test

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            imgs<torch.Tensor>, true_masks<torch.Tensor>, masks_pred1<tuple of torch.Tensors>, masks_pred2<tuple of torch.Tensors> labels<list>, label_names<list>
        """
        # Example #############################################################
        # assert isinstance(batch, dict)
        # assert len(batch) > 0, 'the provided batch is empty'

        # imgs = batch['image']
        # true_masks = batch['mask']
        # labels = batch.get('label', ['']*self.testval_dataloader_kwargs['batch_size'])
        # label_names = batch.get('label_name', ['']*self.testval_dataloader_kwargs['batch_size'])

        # # commenting out main label validation because at level 1
        # # while creating the crops with the desired size some of them
        # # could not have data in the main label
        # # for i in range(labels.shape[0]):
        # #     assert true_masks[i][labels[i]].max() == 1, labels[i].item()

        # if len(imgs.shape) == 5:
        #     # TODO: see how to use and process label_names
        #     imgs, labels, true_masks, _ = self.reshape_data(imgs, labels, true_masks)

        # imgs = imgs.to(device=self.device, dtype=torch.float32)
        # # changing this becaue of the error
        # # RuntimeError: _thnn_conv_depthwise2d_forward not supported on CUDAType for Long
        # # mask_type = torch.float32 if self.module.n_classes == 1 else torch.long
        # # mask_type = torch.float32
        # true_masks = true_masks.to(device=self.device, dtype=torch.float32)

        # with torch.no_grad():
        #     masks_pred1, masks_pred2 = self.model(imgs)

        # # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # # so if we have a tensor then we just put it inside a tuple
        # # to not break the workflow
        # masks_pred1 = masks_pred1 if isinstance(masks_pred1, tuple) else (masks_pred1, )
        # masks_pred2 = masks_pred2 if isinstance(masks_pred2, tuple) else (masks_pred2, )

        # return imgs, true_masks, masks_pred1, masks_pred2, labels, label_names

        raise NotImplementedError("get_validation_data not implemented.")

    def validation_step(self, **kwargs):
        """
        Logic to perform the validation step per batch

        Kwargs:
            batch       <dict>: Dictionary contaning batch data
            testing     <bool>: Whether it is performing testing or validation.
                                Default False
            plot_to_png <bool>: Whether or not save the predictions as PNG files. This option
                                is useful to visually examine the predicted masks.
                                Default False
            mask_plotter <MaskPlotter, None>: Optional MaskPlotter instance.
                                Default None
            imgs_counter <int>: Counter of processed images. Default 0
            apply_threshold <bool>: Whether or not apply thresholding to the predicted mask.
                                Default True

        Returns:
            loss1<torch.Tensor>, loss2<torch.Tensor>,  extra_data<dict>
        """
        # Example #############################################################
        # batch = kwargs.get('batch')
        # testing = kwargs.get('testing', False)
        # plot_to_png = kwargs.get('plot_to_png', False)
        # mask_plotter = kwargs.get('mask_plotter', None)
        # imgs_counter = kwargs.get('imgs_counter', 0)
        # apply_threshold = kwargs.get('apply_threshold', True)

        # assert isinstance(batch, dict), type(batch)
        # assert isinstance(testing, bool), type(testing)
        # assert isinstance(plot_to_png, bool), type(plot_to_png)
        # if mask_plotter:
        #     assert isinstance(mask_plotter, MaskPlotter), type(mask_plotter)
        # assert isinstance(imgs_counter, int), type(imgs_counter)
        # assert isinstance(apply_threshold, bool), type(apply_threshold)

        # loss1 = torch.tensor(0.)
        # loss2 = torch.tensor(0.)

        # with torch.cuda.amp.autocast(enabled=USE_AMP):
        #     imgs, true_masks, masks_pred1, masks_pred2, labels, label_names = self.get_validation_data(batch)

        #     if not testing:
        #         loss1 = torch.sum(torch.stack([
        #             self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred1
        #         ]))
        #         loss2 = torch.sum(torch.stack([
        #             self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred2
        #         ]))

        # # TODO: Try with masks from d5 and other decoders
        # pred1 = masks_pred1[0]  # using mask from decoder d1
        # pred1 = torch.sigmoid(pred1) if self.module.model1.n_classes == 1 else torch.softmax(pred1, dim=1)
        # pred2 = masks_pred2[0]  # using mask from decoder d1
        # pred2 = torch.sigmoid(pred2) if self.module.model2.n_classes == 1 else torch.softmax(pred2, dim=1)

        # if testing and plot_to_png:
        #     # TODO: review if the logic of imgs_counter still works
        #     filenames = tuple(str(imgs_counter + i)+'.model1' for i in range(1, pred1.shape[0]+1))
        #     mask_plotter(imgs, true_masks, pred1, filenames)
        #     filenames = tuple(str(imgs_counter + i)+'.model2' for i in range(1, pred2.shape[0]+1))
        #     mask_plotter(imgs, true_masks, pred2, filenames)

        # if apply_threshold:
        #     # FIXME try calculating the metric without the threshold
        #     pred1 = (pred1 > self.mask_threshold).float()
        #     pred2 = (pred2 > self.mask_threshold).float()

        # self.valid_metrics1.update(pred1, true_masks)
        # self.valid_metrics2.update(pred2, true_masks)

        # extra_data = dict(
        #     imgs=imgs, pred1=pred1, pred2=pred2, true_masks=true_masks, labels=labels,
        #     label_names=label_names
        # )

        # return loss1, loss2, extra_data

        raise NotImplementedError("validation_step not implemented.")

    def validation(self, **kwargs):
        """
        Evaluation using the provided metric. Can be used for validation and testing (with testing=True)

        NOTE: Override this method only if you need a custom logic for the whole validation process

        Kwargs:
            dataloader      <DataLoader>: DataLoader containing the dataset to perform the evaluation
            testing               <bool>: Whether it is performing testing or validation.
                                          Default False
            plot_to_png           <bool>: Whether or not save the predictions as PNG files. This option
                                          is useful to visually examine the predicted masks.
                                          Default False
            saving_dir             <str>: Directory where to save the predictions as PNG files.
                                          If the directory exists, it will be deleted and created again.
                                          Default 'plotted_masks'
            func_plot_palette <callable>: Function to plot and save the colour palette. It must
                                          receive as first argument the saving path. Default None
            plotter_conf          <dict>: initial configuration for MaskPlotter. See
                                          nns.callbacks.plotters.masks import MaskPlotter
                                          Default dict(alpha=.7, dir_per_file=False, superimposed=False, max_values=False, decoupled=False)
        Returns:
            loss1<torch.Tensor>, loss2<torch.Tensor>, metric_scores1<dict>, metric_scores2<dict>,
            extra_data<dict>
        """
        dataloader = kwargs.get('dataloader')
        testing = kwargs.get('testing', False)
        plot_to_png = kwargs.get('plot_to_png', False)
        saving_dir = kwargs.get('saving_dir', 'plotted_masks')
        func_plot_palette = kwargs.get('func_plot_palette', None)
        plotter_conf = kwargs.get(
            'plotter_conf', dict(
                alpha=.7, dir_per_file=False, superimposed=False, max_values=False, decoupled=False)
        )
        plotter_conf['labels_data'] = self.labels_data
        plotter_conf['mask_threshold'] = self.mask_threshold
        # making sure we use the same saving_dir everywhere
        plotter_conf['saving_dir'] = saving_dir

        assert isinstance(dataloader, DataLoader), type(dataloader)
        assert isinstance(testing, bool), type(testing)
        assert isinstance(plot_to_png, bool), type(plot_to_png)
        assert isinstance(saving_dir, str), type(saving_dir)

        if func_plot_palette is not None:
            assert callable(func_plot_palette)

        if testing and plot_to_png:
            clean_create_folder(saving_dir)
            mask_plotter = MaskPlotter(**plotter_conf)
        else:
            mask_plotter = None

        self.model.eval()
        n_val = len(dataloader)  # the number of batchs
        loss1 = loss2 = imgs_counter = 0
        # the folowing variables will store extra data from the last validation batch
        extra_data = None

        for batch in tqdm(dataloader, total=n_val, desc='Testing round', unit='batch', leave=True,
                          disable=not testing or DISABLE_PROGRESS_BAR):
            # TODO: appropriately update validation_step
            loss_1, loss_2, extra_data = self.validation_step(
                batch=batch, testing=testing, plot_to_png=plot_to_png, imgs_counter=imgs_counter,
                mask_plotter=mask_plotter
            )
            loss1 += loss_1
            loss2 += loss_2
            imgs_counter += self.testval_dataloader_kwargs['batch_size']

        # total metrics over all validation batches
        metrics1 = self.valid_metrics1.compute()
        metrics2 = self.valid_metrics2.compute()
        # reset metrics states after each epoch
        self.valid_metrics1.reset()
        self.valid_metrics2.reset()

        if testing and plot_to_png and func_plot_palette is not None:
            func_plot_palette(os.path.join(saving_dir, 'label_palette.png'))

        self.model.train()

        return loss1 / n_val, loss2 / n_val, metrics1, metrics2, extra_data

    def validation_post(self, **kwargs):
        """ Logic to be executed after the validation step """
        pass

    def training_step(self, batch: dict):
        """
        Logic to perform the training step per batch

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            pred1<torch.Tensor>, pred2<torch.Tensor>, true_masks<torch.Tensor>, imgs<torch.Tensor>,
            loss1<torch.Tensor>, loss2<torch.Tensor>, metrics1<dict>, metrics2<dict>, labels<list>,
            label_names<list>
        """
        # Example #############################################################
        # # TODO: part of these lines can be re-used for get_validation_data with minors tweaks
        # #       review if this is a good idea o not
        # imgs = batch['image']
        # true_masks = batch['mask']
        # labels = batch.get('label', ['']*self.train_dataloader_kwargs['batch_size'])
        # label_names = batch.get('label_name', ['']*self.train_dataloader_kwargs['batch_size'])

        # # commenting out main label validation because at level 1
        # # while creating the crops with the desired size some of them
        # # could not have data in the main label
        # # for i in range(labels.shape[0]):
        # #     assert true_masks[i][labels[i]].max() == 1, labels[i].item()
        # if len(imgs.shape) == 5:
        #     # TODO: see how to use and process label_names
        #     imgs, labels, true_masks, _ = self.reshape_data(imgs, labels, true_masks)

        # if imgs.shape[1] != self.module.model1.n_channels:
        #     raise ModelMGRImageChannelsError(self.module.model1.n_channels, imgs.shape[1])
        # if imgs.shape[1] != self.module.model2.n_channels:
        #     raise ModelMGRImageChannelsError(self.module.model2.n_channels, imgs.shape[1])

        # imgs = imgs.to(device=self.device, dtype=torch.float32)
        # # FIXME: review this!!
        # # changing this becaue of the error
        # # RuntimeError: _thnn_conv_depthwise2d_forward not supported on CUDAType
        # #               for Long
        # # mask_type = torch.float32 if self.module.n_classes == 1 else torch.long
        # # mask_type = torch.float32
        # true_masks = true_masks.to(device=self.device, dtype=torch.float32)

        # with torch.cuda.amp.autocast(enabled=USE_AMP):
        #     masks_pred1, masks_pred2 = self.model(imgs)

        #     # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        #     # so if we have a tensor then we just put it inside a tuple
        #     # to not break the workflow
        #     masks_pred1 = masks_pred1 if isinstance(masks_pred1, tuple) else (masks_pred1, )
        #     masks_pred2 = masks_pred2 if isinstance(masks_pred2, tuple) else (masks_pred2, )

        #     loss1 = torch.sum(torch.stack([
        #         self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred1
        #     ]))
        #     loss2 = torch.sum(torch.stack([
        #         self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred2
        #     ]))

        # # using mask from decoder d1
        # # TODO: Try with masks from d5 and other decoders
        # pred1 = masks_pred1[0]
        # pred1 = torch.sigmoid(pred1) if self.module.model1.n_classes == 1 else torch.softmax(pred1, dim=1)
        # pred2 = masks_pred2[0]
        # pred2 = torch.sigmoid(pred2) if self.module.model2.n_classes == 1 else torch.softmax(pred2, dim=1)

        # # FIXME try calculating the metric without the threshold
        # pred1 = (pred1 > self.mask_threshold).float()
        # metrics1 = self.train_metrics1(pred1, true_masks)
        # pred2 = (pred2 > self.mask_threshold).float()
        # metrics2 = self.train_metrics2(pred2, true_masks)

        # return pred1, pred2, true_masks, imgs, loss1, loss2, metrics1, metrics2, labels, label_names

        raise NotImplementedError("training_step not implemented.")

    def training(self):
        """
        Trains the model

        NOTE: Override this method only if you need a custom logic for the whole training process
        """
        global_step = 0
        step_divider = self.n_train // (self.intrain_val * self.train_dataloader_kwargs['batch_size'])
        optimizer1 = self.optimizer1(self.module.model1.parameters(), **self.optimizer1_kwargs)
        optimizer2 = self.optimizer2(self.module.model2.parameters(), **self.optimizer2_kwargs)
        scaler1 = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        scaler2 = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        metric_evaluator = MetricEvaluator(self.metric_mode)
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        early_stopped = False
        checkpoint = Checkpoint(self.checkpoint_interval) if self.checkpoint_interval > 0 else None
        start_epoch = 0
        validation_step = 0
        data_logger = dict(
            train_loss1=[], train_loss2=[], train_metric1=[], train_metric2=[], val_loss1=[], val_loss2=[],
            val_metric1=[], val_metric2=[], lr1=[], lr2=[],
            epoch_train_losses1=[], epoch_train_losses2=[], epoch_train_metrics1=[], epoch_train_metrics2=[],
            epoch_val_losses1=[], epoch_val_losses2=[], epoch_val_metrics1=[], epoch_val_metrics2=[],
            epoch_lr1=[], epoch_lr2=[],
        )
        best_metric = np.NINF
        val_loss_min = np.inf

        # If a checkpoint file is provided, then load it
        if self.ini_checkpoint:
            start_epoch, data_logger = self.load_checkpoint([optimizer1, optimizer2])
            # The losses and metrics are jointly evaluated to correctly decide
            # when the model (model1 and model2) has the best joint results
            val_loss_min = (
                (
                    np.array(data_logger['val_loss1']) + np.array(data_logger['val_loss2'])
                ) / 2
            ).min()
            best_metric = (
                (
                    np.array(self.get_combined_main_metrics(data_logger['val_metric1'])) +
                    np.array(self.get_combined_main_metrics(data_logger['val_metric2']))
                ) / 2
            ).max()
            # increasing to start at the next epoch
            start_epoch += 1

        if self.lr_scheduler1 is not None:
            scheduler1 = self.lr_scheduler1(optimizer1, **self.lr_scheduler1_kwargs)
        else:
            scheduler1 = None

        if self.lr_scheduler2 is not None:
            scheduler2 = self.lr_scheduler2(optimizer2, **self.lr_scheduler2_kwargs)
        else:
            scheduler2 = None

        if self.tensorboard:
            writer = SummaryWriter(
                comment=f'Model1_{self.model1_cls.__name__}_model2_{self.model2_cls.__name__}_LR1_{self.optimizer1_kwargs["lr"]}_LR2_{self.optimizer2_kwargs["lr"]}_BS_{self.train_dataloader_kwargs["batch_size"]}'
            )
        else:
            writer = MagicMock()

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            train_loss1 = train_loss2 = 0

            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img',
                      disable=DISABLE_PROGRESS_BAR) as pbar:
                intrain_chkpt_counter = 0
                intrain_val_counter = 0

                for batch in self.train_loader:
                    pred1, pred2, true_masks, imgs, loss1, loss2, metrics1, metrics2, labels, label_names = \
                        self.training_step(batch)
                    train_loss1 += loss1.item()
                    train_loss2 += loss2.item()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    # loss.backward(retain_graph=True)
                    scaler1.scale(loss1).backward(retain_graph=True)
                    scaler2.scale(loss2).backward(retain_graph=True)
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    # optimizer.step()
                    # TODO: review if using scaler is still a good option for DA cotraining
                    scaler1.step(optimizer1)
                    scaler2.step(optimizer2)
                    scaler1.update()
                    scaler2.update()
                    pbar.update(imgs.shape[0])
                    global_step += 1

                    if global_step % step_divider == 0:
                        validation_step += 1
                        intrain_val_counter += 1
                        val_loss1, val_loss2, val_metrics1, val_metrics2, val_extra_data = self.validation(
                            dataloader=self.val_loader)

                        # maybe if there's no scheduler then the lr shouldn't be plotted
                        writer.add_scalar('learning_rate1', optimizer1.param_groups[0]['lr'], validation_step)
                        writer.add_scalar('learning_rate2', optimizer2.param_groups[0]['lr'], validation_step)
                        data_logger['lr1'].append(optimizer1.param_groups[0]['lr'])
                        data_logger['lr2'].append(optimizer2.param_groups[0]['lr'])
                        writer.add_scalar('Loss/train1', loss1.item(), validation_step)
                        writer.add_scalar('Loss/train2', loss2.item(), validation_step)
                        data_logger['train_loss1'].append(loss1.item())
                        data_logger['train_loss2'].append(loss2.item())

                        for metric_, value_ in chain(metrics1.items(), metrics2.items()):
                            writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                        data_logger['train_metric1'].append(self.prepare_to_save(metrics1))
                        data_logger['train_metric2'].append(self.prepare_to_save(metrics2))
                        writer.add_scalar('Loss/val1', val_loss1.item(), validation_step)
                        writer.add_scalar('Loss/val2', val_loss2.item(), validation_step)
                        data_logger['val_loss1'].append(val_loss1.item())
                        data_logger['val_loss2'].append(val_loss2.item())

                        for metric_, value_ in chain(val_metrics1.items(), val_metrics2.items()):
                            writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                        data_logger['val_metric1'].append(self.prepare_to_save(val_metrics1))
                        data_logger['val_metric2'].append(self.prepare_to_save(val_metrics2))

                        self.print_validation_summary(
                            global_step=global_step, validation_step=validation_step, loss=loss1,
                            metrics=metrics1, val_loss=val_loss1, val_metrics=val_metrics1
                        )
                        self.print_validation_summary(
                            global_step=global_step, validation_step=validation_step, loss=loss2,
                            metrics=metrics2, val_loss=val_loss2, val_metrics=val_metrics2
                        )
                        self.validation_post(
                            pred=pred1, true_masks=true_masks, labels=labels, imgs=imgs,
                            label_names=label_names, writer=writer, validation_step=validation_step,
                            global_step=global_step, val_extra_data=val_extra_data
                        )
                        self.validation_post(
                            pred=pred2, true_masks=true_masks, labels=labels, imgs=imgs,
                            label_names=label_names, writer=writer, validation_step=validation_step,
                            global_step=global_step, val_extra_data=val_extra_data
                        )

                        # TODO: find out if it's better to apply the early stopping to
                        # val_metric or val_loss
                        val_metric = mean([self.get_mean_main_metrics(val_metrics1),
                                           self.get_mean_main_metrics(val_metrics2)])
                        new_val_loss_min = mean([val_loss1.item(), val_loss2.item()])

                        if self.earlystopping_to_metric:
                            if earlystopping(best_metric, val_metric):
                                early_stopped = True
                                break
                        elif earlystopping(new_val_loss_min, val_loss_min):
                            early_stopped = True
                            break

                        if metric_evaluator(val_metric, best_metric):
                            logger.info(
                                f'Mean {self.main_metrics_str} increased'
                                f'({best_metric:.6f} --> {val_metric:.6f}). '
                                'Saving model ...'
                            )
                            self.save()
                            self.save_checkpoint(
                                float(f'{epoch}.{intrain_val_counter}'), [optimizer1, optimizer2],
                                data_logger, best_chkpt=True
                            )
                            best_metric = val_metric

                        if new_val_loss_min < val_loss_min:
                            val_loss_min = new_val_loss_min

                        if self.train_eval_chkpt and checkpoint and checkpoint(epoch):
                            intrain_chkpt_counter += 1
                            self.save_checkpoint(float(f'{epoch}.{intrain_chkpt_counter}'),
                                                 [optimizer1, optimizer2], data_logger)
                        if scheduler1 is not None:
                            # TODO: verify the replacement function is working properly
                            LrShedulerTrack.step(self.lr_scheduler1_track, scheduler1,
                                                 self.get_mean_main_metrics(val_metrics1), val_loss1.item())

                        if scheduler2 is not None:
                            # TODO: verify the replacement function is working properly
                            LrShedulerTrack.step(self.lr_scheduler2_track, scheduler2,
                                                 self.get_mean_main_metrics(val_metrics2), val_loss2.item())

            if self.last_checkpoint:
                self.save_checkpoint(
                    float(f'{epoch}'), [optimizer1, optimizer2], data_logger, last_chkpt=True)

            if early_stopped:
                break

            # computing epoch statistiscs #####################################
            data_logger['epoch_lr1'].append(optimizer1.param_groups[0]['lr'])
            data_logger['epoch_lr2'].append(optimizer2.param_groups[0]['lr'])

            train_batches = len(self.train_loader)
            data_logger['epoch_train_losses1'].append(train_loss1 / train_batches)
            data_logger['epoch_train_losses2'].append(train_loss2 / train_batches)
            # total metrics over all training batches
            data_logger['epoch_train_metrics1'].append(self.prepare_to_save(self.train_metrics1.compute()))
            data_logger['epoch_train_metrics2'].append(self.prepare_to_save(self.train_metrics2.compute()))
            # reset metrics states after each epoch
            self.train_metrics1.reset()
            self.train_metrics2.reset()

            val_loss1, val_loss2,  val_metric1, val_metric2,  _ = self.validation(dataloader=self.val_loader)
            data_logger['epoch_val_losses1'].append(val_loss1.item())
            data_logger['epoch_val_metrics1'].append(self.prepare_to_save(val_metric1))
            data_logger['epoch_val_losses2'].append(val_loss2.item())
            data_logger['epoch_val_metrics2'].append(self.prepare_to_save(val_metric2))

            self.print_epoch_summary(epoch, data_logger)

            if checkpoint and checkpoint(epoch):
                self.save_checkpoint(epoch, [optimizer1, optimizer2], data_logger)

        train_metrics1 = [f'{self.train_prefix1}{metric}' for metric in self.train_metrics1]
        train_metrics2 = [f'{self.train_prefix2}{metric}' for metric in self.train_metrics2]
        val_metrics1 = [f'{self.valid_prefix1}{metric}' for metric in self.valid_metrics1]
        val_metrics2 = [f'{self.valid_prefix2}{metric}' for metric in self.valid_metrics2]

        writer.add_custom_scalars({
            'Metric': {
                'Metric/Train&Val1': ['Multiline', train_metrics1+val_metrics1],
                'Metric/Train&Val2': ['Multiline', train_metrics2+val_metrics2]
            },
            'Loss': {
                'Loss/Train&Val1': ['Multiline', ['Loss/train1', 'Loss/val1']],
                'Loss/Train&Val2': ['Multiline', ['Loss/train2', 'Loss/val2']]
            },
            'LearningRate': {
                'Train1': ['Multiline', ['learning_rate1']],
                'Train2': ['Multiline', ['learning_rate2']]
            }
        })
        writer.close()

        if self.plot_to_disk:
            self.plot_and_save(data_logger, step_divider)

    @timing
    def fit(self):
        """  """
        try:
            self.training()
        except KeyboardInterrupt:
            self.save('INTERRUPTED.pth')
            logger.info("Saved interrupt")

            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @timing
    def test(self, dataloader=None, **kwargs):
        """
        Performs the testing using the provided subdataset

        Args:
            dataloader <DataLoader>: DataLoader containing the dataset to perform the evaluation.
                                     If None the test_loader will be used.
                                     Default None
        """
        if dataloader is None:
            dataloader = self.test_loader

            if self.test_loader is None:
                logger.error("No testing dataloader was provided")
                return
        else:
            assert isinstance(dataloader, DataLoader)

        assert len(dataloader) > 0, "The dataloader did not have any data"

        # Loading the provided checkpoint or the best model obtained during training
        if self.ini_checkpoint:
            _, _ = self.load_checkpoint([
                self.optimizer1(self.module.model1.parameters(), **self.optimizer1_kwargs),
                self.optimizer2(self.module.model2.parameters(), **self.optimizer2_kwargs),
            ])
        else:
            self.load()

        _, metric = self.validation(dataloader=self.test_loader, testing=True, **kwargs)

        logger.info(f'Testing Metric: {metric}')

    def predict_step(self, patch: torch.Tensor):
        """
        Returns the prediction of the patch

        Args:
            patch <torch.Tensor>: patch cropped from the input image
        Returns:
            preds_plus_bg1 <torch.Tensor>, preds_plus_bg2 <torch.Tensor>
        """
        # Example #############################################################
        # TODO: add an example
        raise NotImplementedError("predict_step not implemented.")

    @timing
    def predict(self, image_path: str, read_image: Optional[Callable] = None, /, **kwargs):
        """
        Calculates the masks predictions of the image and saves the outcome as a PNG file

        NOTE: Override this method only if you need a custom logic for the whole process of prediction

        Kwargs:
            image_path        <str>: path to the image to be analyzed
            read_image   <callable>: function to read the image. It must return a
                                     PIL.Image.Image object. By default OpenSlide will be used
                                     to try to open the image (will work for most of WSIs)
            preprocess_image <callable>: function to preprocess the image. By default the basic_preprocess
                                     method is used.
            patch_size        <int>: patch size. Default 640
            patch_overlapping <int>: overlapping between patches. It must be an even number;
                                     furthermore, only the 50% of the overlapping is used to create
                                     final mask. This is necessary to remove inconsistent mask borders
                                     between patches. Default 240
            level             <int>: Image magnification level (when using OpenSlide). Default 2
            alpha           <float>: alpha channel value used when superimposing the mask. Default 0.9
            superimpose      <bool>: If True the superimposed mask is saved, else only the predicted mask.
                                     Default True
            size            <tuple>: Tuple containing the desired final size of the mask. If set to None
                                     the predicted mask will have the same dimensions as the analyzed image.
                                     Default (3000, 3000)
            postprocess_mask <bool>: Whether or not to return only mask predictions from the image foreground.
                                     Default False
            remove_bg_kwargs <dict>: Dictionary of arguments to initialize RemoveBG. It is only used when
                                     postprocess_mask = True. Default {}
        """
        preprocess_image = kwargs.get('preprocess_image', self.basic_preprocess)
        patch_size = kwargs.get('patch_size', 640)
        patch_overlapping = kwargs.get('patch_overlapping', 240)
        level = kwargs.get('level', 2)
        alpha = kwargs.get('alpha', 0.9)
        superimpose = kwargs.get('superimpose', True)
        size = kwargs.get('size', (3000, 3000))
        postprocess_mask = kwargs.get('postprocess_mask', False)
        remove_bg_kwargs = kwargs.get('remove_bg_kwargs', {})

        assert os.path.isfile(image_path), f'{image_path}'
        if read_image:
            assert callable(read_image), 'read_image is not a callable'
        assert callable(preprocess_image), 'preprocess_image is not a callable'
        assert isinstance(patch_size, int), type(patch_size)
        assert patch_size > 0, patch_size
        assert isinstance(patch_overlapping, int), type(patch_overlapping)
        assert patch_overlapping >= 0, patch_overlapping
        assert patch_overlapping % 2 == 0, 'patch_overlapping must be a even number'
        assert isinstance(level, int), type(level)
        assert level >= 0
        assert 0 <= alpha <= 1, alpha
        assert isinstance(superimpose, bool), type(superimpose)
        assert isinstance(postprocess_mask, bool), type(postprocess_mask)
        assert isinstance(remove_bg_kwargs, dict), type(remove_bg_kwargs)

        if size:
            assert isinstance(size, tuple), type(size)
            assert len(size) == 2, len(size)

        # Loading the provided checkpoint or the best model obtained during training
        if self.ini_checkpoint:
            _, _ = self.load_checkpoint([
                self.optimizer1(self.module.model1.parameters(), **self.optimizer1_kwargs),
                self.optimizer2(self.module.model2.parameters(), **self.optimizer2_kwargs)
            ])
        else:
            self.load()

        if read_image:
            img = read_image(image_path)
        else:
            img = ops.open_slide(image_path)
            img = img.read_region(
                (0, 0),
                # ((img.dimensions[0]//2)-200, 0),
                level,
                (int(img.dimensions[0] / (2**level)), int(img.dimensions[1] / (2**level)))
                # ((int(img.dimensions[0] / (2**level))//2)+100, int(img.dimensions[1] / (2**level)))
            )

        assert isinstance(img, Image.Image), type(img)

        if postprocess_mask:
            bg_remover = RemoveBG(**remove_bg_kwargs)
            bg_RGB_mask = bg_remover.get_bg_RGB_mask(img)[1]

        img = np.array(img.convert('RGB')) if img.mode != 'RGB' else np.array(img)
        y_dim, x_dim = img.shape[:2]
        final_mask1 = np.full((3, y_dim, x_dim), 255, dtype='uint8')
        final_mask2 = deepcopy(final_mask1)
        self.model.eval()
        total_x = len(list(get_slices_coords(x_dim, patch_size, patch_overlapping=patch_overlapping)))
        total_y = len(list(get_slices_coords(y_dim, patch_size, patch_overlapping=patch_overlapping)))
        total = total_x * total_y
        half_overlapping = patch_overlapping // 2
        b1 = b2 = c1 = c2 = 0

        logger.info('Making predictions')
        with tqdm(total=total, disable=DISABLE_PROGRESS_BAR) as pbar:
            iy_counter = 0
            for iy in get_slices_coords(y_dim, patch_size, patch_overlapping=patch_overlapping):
                ix_counter = 0
                for ix in get_slices_coords(x_dim, patch_size, patch_overlapping=patch_overlapping):
                    patch = img[iy:iy+patch_size, ix:ix+patch_size]
                    # TODO: the preprocess can also be obtained from the dataloader
                    patch = preprocess_image(patch)
                    patch = torch.from_numpy(patch).type(torch.FloatTensor)
                    patch = torch.unsqueeze(patch, 0)
                    patch = patch.to(device=self.device, dtype=torch.float32)
                    pred1, pred2 = self.predict_step(patch)
                    for pred, final_mask in zip([pred1, pred2], [final_mask1, final_mask2]):
                        rgb = np.full((3, *pred.size()), 255, dtype='uint8')

                        for label in self.labels_data.LABELS:
                            colour_idx = pred == label.id
                            rgb[0][colour_idx] = label.RGB[0]
                            rgb[1][colour_idx] = label.RGB[1]
                            rgb[2][colour_idx] = label.RGB[2]

                        if iy_counter == 0:
                            b1 = 0
                            b2 = half_overlapping
                        elif iy_counter + 1 == total_y:
                            b1 = half_overlapping
                            b2 = 0
                        else:
                            b1 = b2 = half_overlapping

                        if ix_counter == 0:
                            c1 = 0
                            c2 = half_overlapping
                        elif ix_counter + 1 == total_x:
                            c1 = half_overlapping
                            c2 = 0
                        else:
                            c1 = c2 = half_overlapping

                        final_mask[:, iy+b1:iy+patch_size-b2, ix+c1:ix+patch_size -
                                   c2] = rgb[:, b1:patch_size-b2, c1:patch_size-c2]
                    ix_counter += 1
                    pbar.update(1)
                iy_counter += 1

        if superimpose:
            for final_mask, model in zip([final_mask1, final_mask2], ['model1', 'model2']):
                im1 = Image.fromarray(img)
                im2 = Image.fromarray(np.moveaxis(final_mask, [0, 1, 2], [2, 0, 1]))

                if postprocess_mask:
                    im2 = bg_remover(im2, bg_RGB_mask)

                im2 = im2.convert("RGBA")
                new_data = []

                logger.info('Superimposing mask')
                # setting white as transparent
                for item in tqdm(im2.getdata(), disable=DISABLE_PROGRESS_BAR):
                    if item[0] == 255 and item[1] == 255 and item[2] == 255:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)

                im2.putdata(new_data)
                im2_trans = Image.new("RGBA", im2.size)
                im2_trans = Image.blend(im2_trans, im2, alpha)
                im1.paste(im2_trans, (0, 0), im2_trans)

                if size:
                    im1.thumbnail(size, Image.ANTIALIAS)

                im1.save(f'{os.path.basename(image_path)}_{model}.mask.png')
        else:
            for final_mask, model in zip([final_mask1, final_mask2], ['model1', 'model2']):
                final_mask = Image.fromarray(np.moveaxis(final_mask, [0, 1, 2], [2, 0, 1]))

                if size:
                    if postprocess_mask:
                        tmp = bg_remover(final_mask, bg_RGB_mask)

                    tmp.thumbnail(size, Image.ANTIALIAS)
                    final_mask = np.array(tmp)
                else:
                    if postprocess_mask:
                        final_mask = bg_remover(final_mask, bg_RGB_mask)

                    final_mask = np.array(final_mask)

                plt.imsave(
                    f'{os.path.basename(image_path)}_{model}.mask.png',
                    final_mask
                )
