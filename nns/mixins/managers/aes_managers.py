# -*- coding: utf-8 -*-
""" nns/mixins/managers/aes_managers """

import os
import sys
from typing import Union, Optional
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nns.callbacks.metrics import MetricEvaluator
from nns.callbacks.metrics.constants import MetricEvaluatorMode
from nns.callbacks.plotters.masks import MaskPlotter
from nns.mixins.constants import LrShedulerTrack
from nns.mixins.checkpoints import CheckPointMixin
from nns.mixins.data_loggers import DataLoggerMixin
from nns.mixins.sanity_checks import SanityChecksMixin
from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR
from nns.mixins.subdatasets import SubDatasetsMixin
from nns.utils.sync_batchnorm import patch_replication_callback


__all__ = ['AEsModelMGRMixin']


class AEsModelMGRMixin(SanityChecksMixin, CheckPointMixin, DataLoggerMixin, SubDatasetsMixin):
    """
    General segmentation manager mixin for models that use AutoEncoders with their own losses and
    with a single optimizer to update the model weights. All the AEs' losses including the general
    loss are backpropagated individually; then, the model's weights are updated using the unique optimizer.

    Note: the model must return
    (logits<torch.Tensor>, aes_data=Dict[str, namedtuple('Data', ['input', 'output'])])

    Usage:
        class MyModelMGR(AEsModelMGRMixin):
           def get_validation_data(self, batch):
               <write some code>
           def validation_step(self, **kwargs):
               <write some code>
           def validation_post(self, **kwargs):
               <write some code>
           def training_step(self, batch):
               <write some code>
           def predict_step(self, patch):
               <write some code>
           # overwrite as many methods as necessary/required
    """

    def __init__(self, **kwargs):
        """
        Kwargs:
            model (nn.Module): Neuronal network class.
            model_kwargs (dict): Dictionary holding the initial arguments for the model
            cuda (bool): whether or not use cuda
            multigpus <bool>: Set it to True to use several GPUS. It requires to set cuda to True.
                              Default False
            patch_replication_callback <bool>: Whether or not to apply patch_replication_callback. It must
                               be used only if SynchronizedBatchNorm2d has been used. Furthermore, it
                               requires that cuda = multigpus = patch_replication_callback = True and there
                               are multiple GPUs available. Default = False
            epochs (int): number of epochs
            intrain_val <int>: Times to interrupt the iteration over the training dataset
                               to collect statistics, perform validation and update
                               the learning rate. Default 10
            optimizer: optimizer class from torch.optim
            optimizer_kwargs: optimizer keyword arguments
            sanity_checks <bool>: Whether or not run model sanity checks. Default False
            labels_data <object>: class containing all the details of the classes/labels. See
                                   nns.callbacks.plotters.masks.MaskPlotter definition
            data_dimensions <int>: Number of dimension of the data. Currently 2D and 3D supported.
                                   Default 2.
            ###################################################################
            #                         SubDatasetsMixin                        #
            ###################################################################
            dataset (DatasetTemplate): Custom dataset class descendant of gtorch_utils.datasets.segmentation.DatasetTemplate.
                                       See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
            dataset_kwargs (dict): keyword arguments for the dataset
            train_dataloader_kwargs <dict>: Keyword arguments for the train DataLoader
            testval_dataloader_kwargs <dict>: Keyword arguments for the test and validation DataLoaders
            ###################################################################
            lr_scheduler: one learing rate scheuler from torch.optim.lr_scheduler
            lr_scheduler_kwargs (dict): keyword arguments for lr_scheduler_class. Some lr_schedulers
                                        like ReduceLROnPlateau have the option 'mode' to speficy if
                                        the value provided when calling scheduler.step(val_metric)
                                        must be maximized or minimized. If your metric is better when
                                        it is close to 1 then set it to max, otherwise to min. On the
                                        other hand, if you use the loss with the lr_scheduler then you
                                        should set 'mode' to min. If patience is provided, it will be
                                        multiplied by intrain_val to reduce the learning rate after
                                        'patience' epochs
            lr_scheduler_track <bool>: Defines if the lr_scheduler.step must be called with val_loss,
                                       val_metric o no arguments. See nns.mixins.constants.LrShedulerTrack
                                       Default LrShedulerTrack.NO_ARGS
If true it track the loss values, else it tracks the metric values.
                                        Default True
            criterions <list>: List of one of more losses
            mask_threshold <float>: mask threshold. Default 0.5
            metrics    <list>: List of MetricItems to be used by the manager
                               Default [MetricItem(DiceCoefficient(), main=True),]
            metric_mode <int>: Evaluation mode of the metric.
                               See nns.callbacks.metrics.constants.MetricEvaluatorMode
                               Default MetricEvaluatorMode.MAX
            earlystopping_kwargs <dict>: Early stopping parameters. When metric = True, it is applied to the
                                         metric values; otherwise, it is applied to the loss values.
                                         If patience is provided, it will be multiplied by intrain_val to
                                         stop everything after 'patience' epochs.
                                         To disable it just set patience = np.inf
                                         See gtorch_utils.nns.managers.callbacks.EarlyStopping class definition
                                         Default dict(min_delta=1e-3, patience=8, metric=True)
            checkpoint_interval <int>: interval of epochs before saving a checkpoint.
                                  If <= 0 then checkpoints are not saved.
                                  Default 1
            train_eval_chkpt <bool>: If True, a checkpoint will be saved right after each evaluation executed
                                  while processing the training subdataset (e.gl chkpt_1.1.pth.tar)
                                  Default False
            last_checkpoint    <bool>: If True, the last checkpoint will be saved separately. This is
                                       useful when checkpoint_interval is set to zero to save disk
                                       space and you want to have the last checkpoint to get all the
                                       statistics from the whole training process. Default False
            ini_checkpoint <str>: path to checkpoint to load. So the training can continue.
                                  It must be inside the the dir_checkpoints directory.
                                  Default ''
            dir_checkpoint <str>: path to the directory where checkpoints will be saved
            tensorboard <bool>: whether or not plot training data into tensorboard. Default True
            plot_to_disk <bool>: Whether or not plot data training data and save it as images.
                                 Default True
            plot_dir      <str>: Directory where the training plots will be saved. Default 'plots'
        """
        self.model = kwargs.get('model')
        self.model_kwargs = kwargs.get('model_kwargs')
        self.cuda = kwargs.get('cuda', True)
        self.multigpus = kwargs.get('multigpus', False)
        self.patch_replication_callback = kwargs.get('patch_replication_callback', False)
        self.epochs = kwargs.get('epochs', 5)
        self.intrain_val = kwargs.get('intrain_val', 10)
        self.optimizer = kwargs.get('optimizer', torch.optim.RMSprop)
        self.optimizer_kwargs = kwargs.get('optimizer_kwargs', dict(lr=1e-4, weight_decay=1e-8, momentum=.9))
        self.sanity_checks = kwargs.get('sanity_checks', False)
        self.labels_data = kwargs['labels_data']
        self.data_dimensions = kwargs.get('data_dimensions', 2)
        self.dataset = kwargs['dataset']

        self.lr_scheduler = kwargs.get('lr_scheduler', None)
        self.lr_scheduler_kwargs = kwargs.get('lr_scheduler_kwargs', {})
        self.lr_scheduler_track = kwargs.get('lr_scheduler_track', LrShedulerTrack.NO_ARGS)
        self.criterions = kwargs.get('criterions', None)

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

        assert issubclass(self.model, nn.Module), type(self.model)
        assert isinstance(self.model_kwargs, dict), type(self.model_kwargs)
        assert isinstance(self.cuda, bool), type(self.cuda)
        assert isinstance(self.multigpus, bool), type(self.multigpus)
        assert isinstance(self.patch_replication_callback, bool), type(self.patch_replication_callback)
        assert isinstance(self.epochs, int), type(self.epochs)
        assert isinstance(self.intrain_val, int), type(self.intrain_val)
        assert self.intrain_val >= 1, self.intrain_val
        assert isinstance(self.optimizer_kwargs, dict), type(self.optimizer_kwargs)
        assert isinstance(self.data_dimensions, int), type(self.data_dimensions)
        assert self.data_dimensions in (2, 3), 'Only 2D and 3D data is supported'
        assert isinstance(self.lr_scheduler_kwargs, dict), type(self.lr_scheduler_kwargs)
        LrShedulerTrack.validate(self.lr_scheduler_track)
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
        if 'patience' in self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs['patience'] *= self.intrain_val

        if self.plot_to_disk and self.plot_dir and not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.model = self.model(**self.model_kwargs)

        if self.cuda:
            if self.multigpus:
                self.model = nn.DataParallel(self.model)

            if isinstance(self.model, nn.DataParallel):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if torch.cuda.is_available() and torch.cuda.device_count() > 1 and \
                   self.patch_replication_callback:
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

    def __call__(self):
        """ functor call """
        self.fit()
        self.test()

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

    def reshape_data(
            self, imgs: torch.Tensor, labels: Union[torch.Tensor, list],
            true_masks: Optional[torch.Tensor] = None,
            filepaths: Optional[Union[torch.Tensor, list]] = None
    ):
        """
        Reshapes tensors with 2D and 3D data to be properly used
        """
        if self.data_dimensions == 3:
            return self.reshape_3D_data(imgs, labels, true_masks, filepaths)

        return self.reshape_2D_data(imgs, labels, true_masks, filepaths)

    @staticmethod
    def reshape_2D_data(
            imgs: torch.Tensor, labels: Union[torch.Tensor, list],
            true_masks: Optional[torch.Tensor] = None,
            filepaths: Optional[Union[torch.Tensor, list]] = None
    ):
        """
        Reshapes tensors with 2D data to be properly used

        Args:
            imgs             <Tensor>: Tensor containing the images with shape
                                       [batches, crops, channels, height, width] or
                                       [batches, channels, height, width]
            labels     <Tensor, list>: Tensor or list containing the labels
            true_masks       <Tensor>: Tensor containing the ground truth masks with same shape as imgs
            filepaths  <Tensor, list>: Tensor or list containing the filepaths

        Returns:
            imgs <torch.Tensor>, labels<torch.Tensor, list>, true_masks<torch.Tensor, None>,
            filepaths<torch.Tensor, list, None>
        """
        assert torch.is_tensor(imgs)
        assert isinstance(labels, (torch.Tensor, list))
        assert len(imgs.shape) in (5, 4), (
            '2D imgs tensor must be [batches, crops, channels, height, width] or'
            '[batches, channels, height, width]'
        )

        if true_masks is not None:
            assert torch.is_tensor(true_masks)
            assert true_masks.shape[2:] == imgs.shape[2:], 'true_masks and imgs must have the same shape'

            if len(true_masks.shape) == 5:  # [batches, crops, channels, height, width]
                true_masks = true_masks.reshape((-1, *true_masks.shape[2:]))

        if len(imgs.shape) == 5:  # [batches, crops, channels, height, width]
            imgs = imgs.reshape((-1, *imgs.shape[2:]))

        # TODO: determine how to reshape filepaths
        if filepaths is not None:
            assert torch.is_tensor(filepaths) or isinstance(filepaths, list)

            if torch.is_tensor(filepaths):
                filepaths = filepaths.squeeze()

        # TODO: determine how to reshape labels
        if torch.is_tensor(labels):
            labels = labels.squeeze()

        return imgs, labels, true_masks, filepaths

    @staticmethod
    def reshape_3D_data(
            imgs: torch.Tensor, labels: Union[torch.Tensor, list],
            true_masks: Optional[torch.Tensor] = None,
            filepaths: Optional[Union[torch.Tensor, list]] = None
    ):
        """
        Reshapes the tensors with 3D data to be properly used

        Args:
            imgs             <Tensor>: Tensor containing the images with shape
                                       [batches, crops, channels, depth, height, width] or
                                       [batches, channels, depth, height, width]
            labels     <Tensor, list>: Tensor or list containing the labels
            true_masks       <Tensor>: Tensor containing the ground truth masks with same shape as imgs
            filepaths  <Tensor, list>: Tensor or list containing the filepaths

        Returns:
            imgs <torch.Tensor>, labels<torch.Tensor, list>, true_masks<torch.Tensor, None>,
            filepaths<torch.Tensor, list, None>
        """
        assert torch.is_tensor(imgs)
        assert isinstance(labels, (torch.Tensor, list))
        assert len(imgs.shape) in (6, 5), (
            '3D imgs tensor must be [batches, crops, channels, depth, height, width] or'
            '[batches, channels, depth, height, width]'
        )

        if true_masks is not None:
            assert torch.is_tensor(true_masks)
            assert true_masks.shape == imgs.shape, 'imgs and true_masks must have the same shape'

            if len(true_masks.shape) == 6:  # [batches, crops, channels, depth, height, width]
                true_masks = true_masks.reshape((-1, *true_masks.shape[2:]))

        if len(imgs.shape) == 6:  # [batches, crops, channels, depth, height, width]
            imgs = imgs.reshape((-1, *imgs.shape[2:]))

        # TODO: determine how to reshape filepaths
        if filepaths is not None:
            assert torch.is_tensor(filepaths) or isinstance(filepaths, list)

            if torch.is_tensor(filepaths):
                filepaths = filepaths.squeeze()

        # TODO: determine how to reshape labels
        if torch.is_tensor(labels):
            labels = labels.squeeze()

        return imgs, labels, true_masks, filepaths

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

    def get_validation_data(self, batch):
        """
        Returns the data to be used for the validation or test

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            imgs<torch.Tensor>, true_masks<torch.Tensor>, masks_pred<tuple of torch.Tensors>, labels<list>, label_names<list>
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
        #     masks_pred = self.model(imgs)

        # # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # # so if we have a tensor then we just put it inside a tuple
        # # to not break the workflow
        # masks_pred = masks_pred if isinstance(masks_pred, tuple) else (masks_pred, )

        # return imgs, true_masks, masks_pred, labels, label_names
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
            loss<torch.Tensor>,  extra_data<dict>
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

        # loss = torch.tensor(0.)

        # with torch.cuda.amp.autocast(enabled=USE_AMP):
        #     imgs, true_masks, masks_pred, labels, label_names = self.get_validation_data(batch)

        #     if not testing:
        #         loss = torch.sum(torch.stack([
        #             self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred
        #         ]))

        # # TODO: Try with masks from d5 and other decoders
        # pred = masks_pred[0]  # using mask from decoder d1
        # pred = torch.sigmoid(pred) if self.module.n_classes == 1 else torch.softmax(pred, dim=1)

        # if testing and plot_to_png:
        #     # TODO: review if the logic of imgs_counter still works
        #     filenames = tuple(str(imgs_counter + i) for i in range(1, pred.shape[0]+1))
        #     mask_plotter(imgs, true_masks, pred, filenames)

        # if apply_threshold:
        #     # FIXME try calculating the metric without the threshold
        #     pred = (pred > self.mask_threshold).float()

        # self.valid_metrics.update(pred, true_masks)

        # extra_data = dict(imgs=imgs, pred=pred, true_masks=true_masks, labels=labels, label_names=label_names)

        # return loss, extra_data

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
            loss<torch.Tensor>, metric_scores<dict>, extra_data<dict>
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

        if self.sanity_checks:
            self.disable_sanity_checks()
        n_val = len(dataloader)  # the number of batchs
        loss = imgs_counter = 0
        # the folowing variables will store extra data from the last validation batch
        extra_data = None

        for batch in tqdm(dataloader, total=n_val, desc='Testing round', unit='batch', leave=True,
                          disable=not testing or DISABLE_PROGRESS_BAR):
            loss_, extra_data = self.validation_step(
                batch=batch, testing=testing, plot_to_png=plot_to_png, imgs_counter=imgs_counter,
                mask_plotter=mask_plotter
            )
            loss += sum(loss_.values())
            imgs_counter += self.testval_dataloader_kwargs['batch_size']

        # total metrics over all validation batches
        metrics = self.valid_metrics.compute()
        # reset metrics states after each epoch
        self.valid_metrics.reset()

        if testing and plot_to_png and func_plot_palette is not None:
            func_plot_palette(os.path.join(saving_dir, 'label_palette.png'))

        self.model.train()

        if self.sanity_checks:
            self.enable_sanity_checks()

        return loss / n_val, metrics, extra_data

    def validation_post(self, **kwargs):
        """ Logic to be executed after the validation step """
        pass

    def training_step(self, batch):
        """
        Logic to perform the training step per batch

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            pred<torch.Tensor>, true_masks<torch.Tensor>, imgs<torch.Tensor>, loss<torch.Tensor>, metrics<dict>, labels<list>, label_names<list>
        """
        # Example #############################################################
        # assert isinstance(batch, dict), type(batch)
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

        # if imgs.shape[1] != self.module.n_channels:
        #     raise ModelMGRImageChannelsError(self.module.n_channels, imgs.shape[1])

        # imgs = imgs.to(device=self.device, dtype=torch.float32)
        # # FIXME: review this!!
        # # changing this becaue of the error
        # # RuntimeError: _thnn_conv_depthwise2d_forward not supported on CUDAType
        # #               for Long
        # # mask_type = torch.float32 if self.module.n_classes == 1 else torch.long
        # # mask_type = torch.float32
        # true_masks = true_masks.to(device=self.device, dtype=torch.float32)

        # with torch.cuda.amp.autocast(enabled=USE_AMP):
        #     masks_pred = self.model(imgs)

        #     # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        #     # so if we have a tensor then we just put it inside a tuple
        #     # to not break the workflow
        #     masks_pred = masks_pred if isinstance(masks_pred, tuple) else (masks_pred, )

        #     loss = torch.sum(torch.stack([
        #         self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred
        #     ]))

        # # using mask from decoder d1
        # # TODO: Try with masks from d5 and other decoders
        # pred = masks_pred[0]
        # pred = torch.sigmoid(pred) if self.module.n_classes == 1 else torch.softmax(pred, dim=1)

        # # FIXME try calculating the metric without the threshold
        # pred = (pred > self.mask_threshold).float()
        # metrics = self.train_metrics(pred, true_masks)

        # return pred, true_masks, imgs, loss, metrics, labels, label_names
        raise NotImplementedError("training_step not implemented.")

    def training(self):
        """
        Trains the model

        NOTE: Override this method only if you need a custom logic for the whole training process
        """
        global_step = 0
        step_divider = self.n_train // (self.intrain_val * self.train_dataloader_kwargs['batch_size'])
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_kwargs)

        if self.sanity_checks:
            self.add_sanity_checks(optimizer)

        if self.cuda:
            scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        metric_evaluator = MetricEvaluator(self.metric_mode)
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        early_stopped = False
        checkpoint = Checkpoint(self.checkpoint_interval) if self.checkpoint_interval > 0 else None
        start_epoch = 0
        validation_step = 0
        data_logger = dict(
            train_loss=[], train_metric=[], val_loss=[], val_metric=[], lr=[],
            epoch_train_losses=[], epoch_train_metrics=[], epoch_val_losses=[], epoch_val_metrics=[],
            epoch_lr=[],
        )
        best_metric = np.NINF
        val_loss_min = np.inf
        train_batches = len(self.train_loader)

        # If a checkpoint file is provided, then load it
        if self.ini_checkpoint:
            # TODO: once all the losses are settled, load them too!
            start_epoch, data_logger = self.load_checkpoint(optimizer)
            val_loss_min = min(data_logger['val_loss'])
            best_metric = self.get_best_combined_main_metrics(data_logger['val_metric'])
            # increasing to start at the next epoch
            start_epoch += 1

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        else:
            scheduler = None

        if self.tensorboard:
            writer = SummaryWriter(
                comment=f'LR_{self.optimizer_kwargs["lr"]}_BS_{self.train_dataloader_kwargs["batch_size"]}_SCALE_{self.dataset_kwargs.get("img_scale", None)}'
            )
        else:
            writer = MagicMock()

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            epoch_train_loss = 0
            intrain_chkpt_counter = 0
            intrain_val_counter = 0

            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img',
                      disable=DISABLE_PROGRESS_BAR) as pbar:
                for batch in self.train_loader:
                    pred, true_masks, imgs, batch_train_loss, metrics, labels, label_names = \
                        self.training_step(batch)
                    epoch_train_loss += sum([v.item() for v in batch_train_loss.values()])
                    optimizer.zero_grad()

                    if self.cuda:
                        for bt_loss in batch_train_loss.values():
                            scaler.scale(bt_loss).backward(retain_graph=True)
                        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        for bt_loss in batch_train_loss.values():
                            bt_loss.backward(retain_graph=True)
                        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1

                    if global_step % step_divider == 0:
                        validation_step += 1
                        intrain_val_counter += 1
                        # dividing the epoch accummulated train loss by
                        # the number of batches processed so far in the current epoch
                        current_epoch_train_loss = torch.tensor(
                            epoch_train_loss/(intrain_val_counter*step_divider))
                        # printing AE losses
                        for k, v in batch_train_loss.items():
                            print(f'{k}: {v.item()}')
                        ##
                        val_loss, val_metrics, val_extra_data = self.validation(dataloader=self.val_loader)

                        # maybe if there's no scheduler then the lr shouldn't be plotted
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], validation_step)
                        data_logger['lr'].append(optimizer.param_groups[0]['lr'])
                        writer.add_scalar('Loss/train', current_epoch_train_loss.item(), validation_step)
                        data_logger['train_loss'].append(current_epoch_train_loss.item())

                        for metric_, value_ in metrics.items():
                            writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                        data_logger['train_metric'].append(self.prepare_to_save(metrics))
                        writer.add_scalar('Loss/val', val_loss.item(), validation_step)
                        data_logger['val_loss'].append(val_loss.item())

                        for metric_, value_ in val_metrics.items():
                            writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                        data_logger['val_metric'].append(self.prepare_to_save(val_metrics))

                        self.print_validation_summary(
                            global_step=global_step, validation_step=validation_step,
                            loss=current_epoch_train_loss,
                            metrics=metrics, val_loss=val_loss, val_metrics=val_metrics
                        )
                        self.validation_post(
                            pred=pred.detach().cpu(), true_masks=true_masks.detach().cpu(), labels=labels,
                            imgs=imgs.detach().cpu(),
                            label_names=label_names, writer=writer, validation_step=validation_step,
                            global_step=global_step, val_extra_data=val_extra_data
                        )

                        # TODO: find out if it's better to apply the early stopping to
                        # val_metric or val_loss
                        val_metric = self.get_mean_main_metrics(val_metrics)

                        if self.earlystopping_to_metric:
                            if earlystopping(best_metric, val_metric):
                                early_stopped = True
                                break
                        elif earlystopping(val_loss.item(), val_loss_min):
                            early_stopped = True
                            break

                        if metric_evaluator(val_metric, best_metric):
                            logger.info(
                                f'{self.main_metrics_str} increased'
                                f'({best_metric:.6f} --> {val_metric:.6f}). '
                                'Saving model ...'
                            )
                            self.save()
                            self.save_checkpoint(
                                float(f'{epoch}.{intrain_val_counter}'), optimizer, data_logger,
                                best_chkpt=True
                            )
                            best_metric = val_metric

                        if val_loss.item() < val_loss_min:
                            val_loss_min = val_loss.item()

                        if self.train_eval_chkpt and checkpoint and checkpoint(epoch):
                            intrain_chkpt_counter += 1
                            self.save_checkpoint(float(f'{epoch}.{intrain_chkpt_counter}'), optimizer, data_logger)
                        if scheduler is not None:
                            # TODO: verify the replacement function is working properly
                            LrShedulerTrack.step(self.lr_scheduler_track, scheduler, val_metric, val_loss)

            # computing epoch statistiscs #####################################
            data_logger['epoch_lr'].append(optimizer.param_groups[0]['lr'])
            data_logger['epoch_train_losses'].append(epoch_train_loss / train_batches)
            # total metrics over all training batches
            data_logger['epoch_train_metrics'].append(self.prepare_to_save(self.train_metrics.compute()))
            # reset metrics states after each epoch
            self.train_metrics.reset()

            val_loss, val_metric, _ = self.validation(dataloader=self.val_loader)
            data_logger['epoch_val_losses'].append(val_loss.item())
            data_logger['epoch_val_metrics'].append(self.prepare_to_save(val_metric))

            self.print_epoch_summary(epoch, data_logger)

            if checkpoint and checkpoint(epoch):
                self.save_checkpoint(epoch, optimizer, data_logger)

            if self.last_checkpoint:
                self.save_checkpoint(float(f'{epoch}'), optimizer, data_logger, last_chkpt=True)

            if early_stopped:
                break

        train_metrics = [f'{self.train_prefix}{metric}' for metric in self.train_metrics]
        val_metrics = [f'{self.valid_prefix}{metric}' for metric in self.valid_metrics]

        writer.add_custom_scalars({
            'Metric': {'Metric/Train&Val': ['Multiline', train_metrics+val_metrics]},
            'Loss': {'Loss/Train&Val': ['Multiline', ['Loss/train', 'Loss/val']]},
            'LearningRate': {'Train': ['Multiline', ['learning_rate']]}
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
            _, _ = self.load_checkpoint(self.optimizer(self.model.parameters(), **self.optimizer_kwargs))
        else:
            self.load()

        _, metric, _ = self.validation(dataloader=self.test_loader, testing=True, **kwargs)

        logger.info(f'Testing Metric: {metric["val_DiceCoefficient"].item()}')

    def predict_step(self, patch):
        """
        Returns the prediction of the patch

        Args:
            patch <torch.Tensor>: patch cropped from the input image
        Returns:
            preds_plus_bg <torch.Tensor>
        """
        # Example #############################################################
        # assert isinstance(patch, torch.Tensor), type(patch)

        # with torch.no_grad():
        #     preds = self.model(patch)

        # # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # # so we take the predictions from d1
        # # TODO: Try with masks from d5 and other decoders
        # preds = preds[0] if isinstance(preds, tuple) else preds
        # preds = preds[0]  # using masks from the only batch returned
        # preds = torch.sigmoid(preds) if self.module.n_classes == 1 else torch.softmax(preds, dim=0)

        # # adding an extra class full of zeros to represent anything else than the
        # # defined classes like background or any other not identified thing
        # preds_plus_bg = torch.cat([
        #     torch.zeros((1, *preds.shape[1:])), preds.cpu()], dim=0)
        # preds_plus_bg[preds_plus_bg <= self.mask_threshold] = 0
        # # preds_plus_bg[preds_plus_bg <= 0] = 0
        # preds_plus_bg = torch.argmax(preds_plus_bg, dim=0)

        # return preds_plus_bg

        raise NotImplementedError("predict_step not implemented.")

    @timing
    def predict(self, image_path, read_image=None, /, **kwargs):
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
            _, _ = self.load_checkpoint(self.optimizer(self.model.parameters(), **self.optimizer_kwargs))
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
        final_mask = np.full((3, y_dim, x_dim), 255, dtype='uint8')
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
                    pred = self.predict_step(patch)
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

                    final_mask[:, iy+b1:iy+patch_size-b2, ix+c1:ix+patch_size-c2] = \
                        rgb[:, b1:patch_size-b2, c1:patch_size-c2]
                    ix_counter += 1
                    pbar.update(1)
                iy_counter += 1

        if superimpose:
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

            im1.save(f'{os.path.basename(image_path)}.mask.png')
        else:
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
                f'{os.path.basename(image_path)}.mask.png',
                final_mask
            )
