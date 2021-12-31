# -*- coding: utf-8 -*-
""" nns/mixins/managers """

import os
import sys
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import openslide as ops
import torch
from gtorch_utils.nns.managers.callbacks import Checkpoint, EarlyStopping
from gtorch_utils.segmentation.metrics import dice_coeff
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
from nns.mixins.subdatasets import SubDatasetsMixin


class ModelMGRMixin(CheckPointMixin, DataLoggerMixin, SubDatasetsMixin):
    """
    General segmentation model manager

    Usage:
        class MyModelMGR(ModelMGRMixin):
           ...

        model = MyModelMGR(
            model=UNet(n_channels=3, n_classes=10, bilinear=True),
            cuda=True,
            epochs=10,
            intrain_val=2,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
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
            metric=DC_RNPV(dice_threshold=0.25, always_conditioned=True),   # dice_coeff,
            metric_mode=MetricEvaluatorMode.MAX,
            earlystopping_kwargs=dict(min_delta=1e-3, patience=np.inf, metric=True),
            checkpoint_interval=1,
            train_eval_chkpt=True,
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
            model (nn.Module, nn.DataParallel): Neuronal network instance. If going to use
                               multiple gpus, then just wrap it with nn.DataParallel and
                               set cuda=True
            logits (bool): Set it to True if the model returns logits. Default False
            sigmoid (bool): If True sigmoid will be applied to NN output, else softmax.
                            Only used when logits = True. Default True
            cuda (bool): whether or not use cuda
            epochs (int): number of epochs
            intrain_val <int>: Times to interrupt the iteration over the training dataset
                               to collect statistics, perform validation and update
                               the learning rate. The equation is:
                               global_step % (n_train // (intrain_val * batch_size)) == 0
            optimizer: optimizer class from torch.optim
            optimizer_kwargs: optimizer keyword arguments
            labels_data <object>: class containing all the details of the classes/labels. See
                                   nns.callbacks.plotters.masks.MaskPlotter definition
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
                                        should set 'mode' to min.
            lr_scheduler_track <bool>: Defines if the lr_scheduler.step must be called with val_loss,
                                       val_metric o no arguments. See nns.mixins.constants.LrShedulerTrack
                                       Default LrShedulerTrack.NO_ARGS
If true it track the loss values, else it tracks the metric values.
                                        Default True
            criterions <list>: List of one of more losses
            mask_threshold <float>: mask threshold. Default 0.5
            metric <callable>: metric to be used to measure the quality of predicted masks.
                               Default dice_coeff
            metric_mode <int>: Evaluation mode of the metric.
                               See nns.callbacks.metrics.constants.MetricEvaluatorMode
                               Default MetricEvaluatorMode.MAX
            earlystopping_kwargs (dict): Early stopping parameters. When metric = True, it is applied to the
                                         metric values; otherwise, it is applied to the loss values.
                                         To disable it just set patience = np.inf
                                         See gtorch_utils.nns.managers.callbacks.EarlyStopping class definition
                                         Default dict(min_delta=1e-3, patience=8, metric=True)
            checkpoint_interval <int>: interval of epochs before saving a checkpoint.
                                  If <= 0 then checkpoints are not saved.
                                  Default 1
            train_eval_chkpt <bool>: If True, a checkpoint will be saved right after each evaluation executed
                                  while processing the training subdataset (e.gl chkpt_1.1.pth.tar)
                                  Default False
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
        self.logits = kwargs.get('logits', False)
        self.sigmoid = kwargs.get('sigmoid', True)
        self.cuda = kwargs.get('cuda', True)
        self.epochs = kwargs.get('epochs', 5)
        self.intrain_val = kwargs.get('intrain_val', 10)
        self.optimizer = kwargs.get('optimizer', torch.optim.RMSprop)
        self.optimizer_kwargs = kwargs.get('optimizer_kwargs', dict(lr=1e-4, weight_decay=1e-8, momentum=.9))
        self.labels_data = kwargs['labels_data']
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
        self.metric = kwargs.get('metric', dice_coeff)
        self.metric_mode = kwargs.get('metric_mode', MetricEvaluatorMode.MAX)
        self.earlystopping_kwargs = kwargs.get(
            'earlystopping_kwargs', dict(min_delta=1e-3, patience=8, metric=True))
        self.earlystopping_to_metric = self.earlystopping_kwargs.pop('metric')
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 1)
        self.train_eval_chkpt = kwargs.get('train_eval_chkpt', False)
        self.ini_checkpoint = kwargs.get('ini_checkpoint', '')
        self.dir_checkpoints = kwargs.get('dir_checkpoints', 'checkpoints')
        self.tensorboard = kwargs.get('tensorboard', True)
        self.plot_to_disk = kwargs.get('plot_to_disk', True)
        self.plot_dir = kwargs.get('plot_dir', 'plots')

        assert isinstance(self.model, (nn.Module, nn.DataParallel)), type(self.model)
        assert isinstance(self.logits, bool), type(self.logits)
        assert isinstance(self.sigmoid, bool), type(self.sigmoid)
        assert isinstance(self.cuda, bool), type(self.cuda)
        assert isinstance(self.epochs, int), type(self.epochs)
        assert isinstance(self.intrain_val, int), type(self.intrain_val)
        assert isinstance(self.optimizer_kwargs, dict), type(self.optimizer_kwargs)
        assert isinstance(self.lr_scheduler_kwargs, dict), type(self.lr_scheduler_kwargs)
        LrShedulerTrack.validate(self.lr_scheduler_track)
        assert isinstance(self.criterions, list), type(self.criterions)
        assert isinstance(self.mask_threshold, float), type(self.mask_threshold)
        assert callable(self.metric), 'metric must be a callable'
        MetricEvaluatorMode.validate(self.metric_mode)
        assert isinstance(self.earlystopping_kwargs, dict), type(self.earlystopping_kwargs)
        assert isinstance(self.checkpoint_interval, int), type(self.checkpoint_interval)
        assert isinstance(self.train_eval_chkpt, bool), type(self.train_eval_chkpt)
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

        if self.plot_to_disk and self.plot_dir and not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        if self.cuda:
            if isinstance(self.model, nn.DataParallel):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            if self.device == "cpu":
                logger.warning("CUDA is not available. Using CPU")
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.init_SubDatasetsMixin(**kwargs)

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
            imgs<torch.tensor>, true_masks<torch.tensor>, masks_pred<tuple of torch.Tensors>, labels<list>, label_names<list>
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
            batch <>:
            testing <bool>:
            plot_to_png <bool>:
            mask_plotter <>:

        Returns:
            loss, metric, imgs_counter
        """
        # Example #############################################################
        # batch = kwargs.get('batch')
        # testing = kwargs.get('testing')
        # plot_to_png = kwargs.get('plot_to_png')
        # mask_plotter = kwargs.get('mask_plotter')
        # loss = imgs_counter = metric = 0

        # imgs, true_masks, masks_pred = self.get_validation_data(batch)

        # masks_pred has predictions from 5 decoders, so we chose to use masks from
        # decoder d1
        # pred = masks_pred[0]  # using mask from decoder d1

        # if not testing:
        #     loss += torch.sum(torch.stack([
        #         self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred
        #     ]))

        # if self.logits:
        #     pred = F.sigmoid(pred) if self.sigmoid else F.softmax(pred, dim=1)

        # if testing and plot_to_png:
        #     filenames = tuple(str(imgs_counter + i) for i in range(1, pred.shape[0]+1))
        #     imgs_counter += pred.shape[0]
        #     mask_plotter(imgs, true_masks, pred, filenames)

        # # FIXME try calculating the metric without the threshold
        # pred = (pred > self.mask_threshold).float()
        # metric += self.metric(pred, true_masks).item()

        # return loss, metric, imgs_counter

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
            loss<torch.Tensor>, metric_score<float>, extra_data<dict>
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
        # FIXME: review this!!
        n_val = len(dataloader)  # the number of batchs
        loss = 0
        metric = 0
        imgs_counter = 0
        # the folowing variables will store extra data from the last validation batch
        extra_data = None

        for batch in tqdm(dataloader, total=n_val, desc='Testing round', unit='batch', leave=True, disable=not testing):
            loss_, metric_, imgs_counter_, extra_data = self.validation_step(
                batch=batch, testing=testing, loss=loss, plot_to_png=plot_to_png,
                imgs_counter=imgs_counter, mask_plotter=mask_plotter, metric=metric
            )
            loss += loss_
            metric += metric_
            imgs_counter += imgs_counter_

        if testing and plot_to_png and func_plot_palette is not None:
            func_plot_palette(os.path.join(saving_dir, 'label_palette.png'))

        self.model.train()

        return loss / n_val,  metric / n_val, extra_data

    def validation_post(self, **kwargs):
        """ Logic to be executed after the validation step """
        pass

    def training_step(self, batch):
        """
        Logic to perform the training step per batch

        Args:
            batch            <dict>: Dictionary contaning batch data

        Returns:
            pred<torch.Tensor>, true_masks<torch.Tensor>, imgs<torch.Tensor>, loss<torch.Tensor>, metric<float>, labels<list>, label_names<list>
        """
        # Example #############################################################
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
        # # mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
        # # mask_type = torch.float32
        # true_masks = true_masks.to(device=self.device, dtype=torch.float32)
        # masks_pred = self.model(imgs)

        # # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # # so if we have a tensor then we just put it inside a tuple
        # # to not break the workflow
        # masks_pred = masks_pred if isinstance(masks_pred, tuple) else (masks_pred, )

        # # NOTE: FOR CROSSENTROPYLOSS I SHOULD BE USING the LOGITS ....
        # # summing the losses from the outcome(s) and then backpropagating it
        # loss = torch.sum(torch.stack([
        #     self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred
        # ]))

        # # using mask from decoder d1
        # # TODO: Try with masks from d5 and other decoders
        # pred = masks_pred[0]
        # # TODO: Review the returned types and update the docstring
        # __import__("pdb").set_trace()

        # if self.logits:
        #     pred = F.sigmoid(pred) if self.sigmoid else F.softmax(pred, dim=1)

        # # FIXME try calculating the metric without the threshold
        # pred = (pred > self.mask_threshold).float()
        # metric = self.metric(pred, true_masks).item()

        # return pred, true_masks, imgs, loss, metric, labels, label_names

        raise NotImplementedError("training_step not implemented.")

    def training(self):
        """
        Trains the model

        NOTE: Override this method only if you need a custom logic for the whole training process
        """
        global_step = 0
        step_divider = self.n_train // (self.intrain_val * self.train_dataloader_kwargs['batch_size'])
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_kwargs)
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

        # If a checkpoint file is provided, then load it
        if self.ini_checkpoint:
            # TODO: once all the losses are settled, load them too!
            start_epoch, data_logger = self.load_checkpoint(optimizer)
            val_loss_min = min(data_logger['val_loss'])
            best_metric = max(data_logger['val_metric'])
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
            train_loss = 0
            train_metric = 0

            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img') as pbar:
                batch_eval_counter = 0

                for batch in self.train_loader:
                    pred, true_masks, imgs, loss, metric, labels, label_names = self.training_step(batch)
                    train_loss += loss.item()
                    train_metric += metric
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    optimizer.step()
                    pbar.update(imgs.shape[0])
                    global_step += 1

                    if global_step % step_divider == 0:
                        validation_step += 1
                        val_loss, val_metric, val_extra_data = self.validation(dataloader=self.val_loader)
                        val_loss_min = min(val_loss.item(), val_loss_min)

                        # maybe if there's no scheduler then the lr shouldn't be plotted
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], validation_step)
                        data_logger['lr'].append(optimizer.param_groups[0]['lr'])
                        writer.add_scalar('Loss/train', loss.item(), validation_step)
                        data_logger['train_loss'].append(loss.item())
                        writer.add_scalar('Metric/train', metric, validation_step)
                        data_logger['train_metric'].append(metric)
                        writer.add_scalar('Loss/val', val_loss.item(), validation_step)
                        data_logger['val_loss'].append(val_loss.item())
                        writer.add_scalar('Metric/val', val_metric, validation_step)
                        data_logger['val_metric'].append(val_metric)
                        logger.info(
                            f'Global batch: {global_step} \t Validation batch {validation_step} \t'
                            f'Train loss: {loss.item():.6f} \t Train metric: {metric:.6f} \t'
                            f'Val loss: {val_loss.item():.6f} \t Val metric: {val_metric:.6f}'
                        )

                        self.validation_post(
                            pred=pred, true_masks=true_masks, labels=labels, imgs=imgs,
                            label_names=label_names, writer=writer, validation_step=validation_step,
                            global_step=global_step, val_extra_data=val_extra_data
                        )

                        # TODO: find out if it's better to apply the early stopping to
                        # val_metric or val_loss
                        if self.earlystopping_to_metric:
                            if earlystopping(best_metric, val_metric):
                                early_stopped = True
                                break
                        elif earlystopping(val_loss.item(), val_loss_min):
                            early_stopped = True
                            break

                        if metric_evaluator(val_metric, best_metric):
                            logger.info(
                                f'Metric increased ({best_metric:.6f} --> {val_metric:.6f}).'
                                ' Saving model ...'
                            )
                            self.save()
                            best_metric = val_metric

                        if self.train_eval_chkpt and checkpoint and checkpoint(epoch):
                            batch_eval_counter += 1
                            self.save_checkpoint(float(f'{epoch}.{batch_eval_counter}'), optimizer, data_logger)

                        # TODO: find out if it's better to apply the early stopping to
                        # val_metric or val_loss. In train.py the val_metric was used...
                        if scheduler is not None:
                            # TODO: verify the replacemente function is working properly
                            LrShedulerTrack.step(self.lr_scheduler_track, scheduler, val_metric, val_loss)

            if early_stopped:
                break

            # computing epoch statistiscs #####################################
            data_logger['epoch_lr'].append(optimizer.param_groups[0]['lr'])

            train_batches = len(self.train_loader)
            data_logger['epoch_train_losses'].append(train_loss / train_batches)
            data_logger['epoch_train_metrics'].append(train_metric / train_batches)

            val_loss, val_metric, _ = self.validation(dataloader=self.val_loader)
            data_logger['epoch_val_losses'].append(val_loss)
            data_logger['epoch_val_metrics'].append(val_metric)

            logger.info(
                f'Epoch {epoch+1} train loss: {data_logger["epoch_train_losses"][epoch]:.6f} '
                f'\tval loss: {val_loss:.6f}'
            )
            logger.info(
                f'Epoch {epoch+1} train metric: {data_logger["epoch_train_metrics"][epoch]:.6f} '
                f'\tval metric: {val_metric:.6f}'
            )

            if checkpoint and checkpoint(epoch):
                self.save_checkpoint(epoch, optimizer, data_logger)

        writer.add_custom_scalars({
            'Metric': {'Metric/Train&Val': ['Multiline', ['Metric/train', 'Metric/val']]
                       },
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

        _, metric = self.validation(dataloader=self.test_loader, testing=True, **kwargs)

        logger.info(f'Testing Metric: {metric}')

    def predict_step(self, patch):
        """
        Returns the prediction of the patch

        Args:
            patch <np.ndarray>: patch cropped from the input image
        Returns:
            preds_plus_bg <torch.tensor>
        """
        # Example #############################################################
        # assert isinstance(patch, np.ndarray), type(patch)

        # with torch.no_grad():
        #     preds = self.model(patch)

        # # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # # so we take the predictions from d1
        # # TODO: Try with masks from d5 and other decoders
        # preds = preds[0] if isinstance(preds, tuple) else preds
        # preds = preds[0]  # using masks from the only batch returned

        # if self.logits:
        #     preds = F.sigmoid(preds) if self.sigmoid else F.softmax(preds, dim=0)

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
    def predict(self, **kwargs):
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
            level             <int>: image magnification level. Default 2
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
        image_path = kwargs.get('image_path')
        read_image = kwargs.get('read_image', None)
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
        with tqdm(total=total) as pbar:
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
            for item in tqdm(im2.getdata()):
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
