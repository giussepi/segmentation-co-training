# -*- coding: utf-8 -*-
""" nns/mixins/managers/multipreds_managers """


import os
from unittest.mock import MagicMock

import numpy as np
import torch
from gtorch_utils.nns.managers.callbacks import Checkpoint, EarlyStopping
from gutils.decorators import timing
from gutils.folders import clean_create_folder
from logzero import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nns.callbacks.metrics import MetricEvaluator
from nns.callbacks.plotters.masks import MaskPlotter
from nns.mixins.constants import LrShedulerTrack
from nns.mixins.managers.base import BaseModelMGR
from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR
from nns.mixins.torchmetrics.mixins import ModularTorchMetricsMixin


__all__ = ['MultiPredsModelMGRMixin']


class MultiPredsModelMGRMixin(ModularTorchMetricsMixin, BaseModelMGR):
    """
    General segmentation manager mixin for models that returns multiple masks/logits and use an single
    optimizers. All the losses from the masks are added together and updates the weights using
    one optimizer.

    Note: The model passed to the manager must have a module_names attribute containing the some names
    representing the output masks. Thus, if the model returns 4 prediction masks, then its module_names
    attribute should have 4 names e.g. self.module_names = ['de1', 'de2', 'd3', 'd4']

    Usage:
        class MyModelMGR(MultiPredsModelMGRMixin):
           ...

    """

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
            losses<List[torch.Tensor]>, metric_scores<List[dict]>, extra_data<dict>
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
        imgs_counter = 0
        losses = None
        # the folowing variables will store extra data from the last validation batch
        extra_data = None

        for batch in tqdm(dataloader, total=n_val, desc='Testing round', unit='batch', leave=True,
                          disable=not testing or DISABLE_PROGRESS_BAR):
            loss_, extra_data = self.validation_step(
                batch=batch, testing=testing, plot_to_png=plot_to_png, imgs_counter=imgs_counter,
                mask_plotter=mask_plotter
            )

            if losses:
                for idx, _ in enumerate(losses):
                    losses[idx] += loss_[idx]
            else:
                losses = loss_

            imgs_counter += self.testval_dataloader_kwargs['batch_size']

        # total metrics over all validation batches
        metrics = []

        for idx, _ in enumerate(self.module.module_names, start=1):
            metrics.append(getattr(self, f'valid_metrics{idx}').compute())
            # reset metrics states after each epoch
            getattr(self, f'valid_metrics{idx}').reset()

        if testing and plot_to_png and func_plot_palette is not None:
            func_plot_palette(os.path.join(saving_dir, 'label_palette.png'))

        self.model.train()

        if self.sanity_checks:
            self.enable_sanity_checks()

        for loss in losses:
            loss /= n_val

        return losses, metrics, extra_data

    def training(self):
        """
        Trains the model
        """
        global_step = 0
        step_divider = self.n_train // (self.intrain_val * self.train_dataloader_kwargs['batch_size'])
        metric_evaluator = MetricEvaluator(self.metric_mode)
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        early_stopped = False
        checkpoint = Checkpoint(self.checkpoint_interval) if self.checkpoint_interval > 0 else None
        start_epoch = 0
        validation_step = 0
        modules = self.module.module_names
        data_logger = dict(lr=[], epoch_lr=[])

        for idx, _ in enumerate(modules, start=1):
            data_logger[f'train_loss{idx}'] = []
            data_logger[f'train_metric{idx}'] = []
            data_logger[f'val_loss{idx}'] = []
            data_logger[f'val_metric{idx}'] = []
            data_logger[f'epoch_train_losses{idx}'] = []
            data_logger[f'epoch_train_metrics{idx}'] = []
            data_logger[f'epoch_val_losses{idx}'] = []
            data_logger[f'epoch_val_metrics{idx}'] = []

        best_metric = np.NINF
        val_loss_min = np.inf
        train_batches = len(self.train_loader)
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_kwargs)
        # If a checkpoint file is provided, then load it
        if self.ini_checkpoint:
            start_epoch, data_logger = self.load_checkpoint(optimizer)
            val_loss_min = min(data_logger[f'val_loss{len(modules)+1}'])
            best_metric = self.get_best_combined_main_metrics(data_logger[f'val_metric{len(modules)+1}'])
            # increasing to start at the next epoch
            start_epoch += 1

        if self.sanity_checks:
            self.add_sanity_checks(optimizer)

        if self.cuda:
            scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

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
            epoch_train_loss = torch.zeros(len(modules)).to(self.device)
            intrain_chkpt_counter = 0
            intrain_val_counter = 0

            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img',
                      disable=DISABLE_PROGRESS_BAR) as pbar:
                for batch in self.train_loader:
                    pred, true_masks, imgs, batch_train_loss, metrics, labels, label_names = \
                        self.training_step(batch)

                    for idx, _ in enumerate(modules):
                        epoch_train_loss[idx] += batch_train_loss[idx].item()

                    optimizer.zero_grad()

                    if self.cuda:
                        scaler.scale(sum(batch_train_loss)).backward(retain_graph=True)
                        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        sum(batch_train_loss).backward(retain_graph=True)
                        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                        optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1

                    if global_step % step_divider == 0:
                        validation_step += 1
                        intrain_val_counter += 1
                        # dividing the epoch accummulated train loss by
                        # the number of batches processed so far in the current epoch
                        current_epoch_train_loss = epoch_train_loss/(intrain_val_counter*step_divider)
                        val_loss, val_metrics, val_extra_data = self.validation(dataloader=self.val_loader)

                        # maybe if there's no scheduler then the lr shouldn't be plotted
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], validation_step)
                        data_logger['lr'].append(optimizer.param_groups[0]['lr'])

                        for idx, _ in enumerate(modules, start=1):
                            writer.add_scalar(f'Loss{idx}/train',
                                              current_epoch_train_loss[idx-1].item(), validation_step)
                            data_logger[f'train_loss{idx}'].append(current_epoch_train_loss[idx-1].item())

                            for metric_, value_ in metrics[idx-1].items():
                                writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                            data_logger[f'train_metric{idx}'].append(self.prepare_to_save(metrics[idx-1]))
                            writer.add_scalar(f'Loss{idx}/val', val_loss[idx-1].item(), validation_step)
                            data_logger[f'val_loss{idx}'].append(val_loss[idx-1].item())

                            for metric_, value_ in val_metrics[idx-1].items():
                                writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                            data_logger[f'val_metric{idx}'].append(self.prepare_to_save(val_metrics[idx-1]))

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

                        # NOTE: the following processes will be led by the last mask/pred
                        # TODO: find out if it's better to apply the early stopping to
                        # val_metric or val_loss
                        val_metric = [0]*len(modules)

                        for idx, metrics in enumerate(val_metrics):
                            val_metric[idx] = self.get_mean_main_metrics(metrics)

                        if self.earlystopping_to_metric:
                            if earlystopping(best_metric, val_metric[-1]):
                                early_stopped = True
                                break
                        elif earlystopping(val_loss[-1].item(), val_loss_min):
                            early_stopped = True
                            break

                        if metric_evaluator(val_metric[-1], best_metric):
                            logger.info(
                                f'{self.main_metrics_str} increased'
                                f'({best_metric:.6f} --> {val_metric[-1]:.6f}). '
                                'Saving model ...'
                            )
                            self.save()
                            self.save_checkpoint(
                                float(f'{epoch}.{intrain_val_counter}'), optimizer, data_logger,
                                best_chkpt=True
                            )
                            best_metric = val_metric[-1]

                        val_loss_min = min(val_loss_min, val_loss[-1].item())

                        if self.train_eval_chkpt and checkpoint and checkpoint(epoch):
                            intrain_chkpt_counter += 1
                            self.save_checkpoint(float(f'{epoch}.{intrain_chkpt_counter}'), optimizer, data_logger)
                        if scheduler is not None:
                            # TODO: verify the replacement function is working properly
                            LrShedulerTrack.step(self.lr_scheduler_track, scheduler, val_metric[-1], val_loss[-1])

            # computing epoch statistiscs #####################################
            data_logger['epoch_lr'].append(optimizer.param_groups[0]['lr'])

            for idx in range(len(modules)):
                data_logger[f'epoch_train_losses{idx+1}'].append(epoch_train_loss[idx].item() / train_batches)
                # total metrics over all training batches
                data_logger[f'epoch_train_metrics{idx+1}'].append(
                    self.prepare_to_save(getattr(self, f'train_metrics{idx+1}').compute()))
                # reset metrics states after each epoch
                getattr(self, f'train_metrics{idx+1}').reset()

            val_loss, val_metric, _ = self.validation(dataloader=self.val_loader)

            for idx in range(len(modules)):
                data_logger[f'epoch_val_losses{idx+1}'].append(val_loss[idx].item())
                data_logger[f'epoch_val_metrics{idx+1}'].append(self.prepare_to_save(val_metric[idx]))

            self.print_epoch_summary(epoch, data_logger)

            if checkpoint and checkpoint(epoch):
                self.save_checkpoint(epoch, optimizer, data_logger)

            if self.last_checkpoint:
                self.save_checkpoint(float(f'{epoch}'), optimizer, data_logger, last_chkpt=True)

            if early_stopped:
                break

        train_val_sets = []
        loss_sets = []

        for idx, _ in enumerate(modules, start=1):
            train_val_sets.append(
                [getattr(self, f'train_prefix{idx}')+f'{metric}'
                 for metric in getattr(self, f'train_metrics{idx}')] +
                [getattr(self, f'valid_prefix{idx}')+f'{metric}'
                 for metric in getattr(self, f'valid_metrics{idx}')]
            )
            loss_sets.append([f'Loss{idx}/train', 'Loss{idx}/val'])

        writer.add_custom_scalars({
            'Metric': {
                'Metric1/Train&Val': ['Multiline', train_val_sets[0]],
                'Metric2/Train&Val': ['Multiline', train_val_sets[1]],
                'Metric3/Train&Val': ['Multiline', train_val_sets[2]],
                'Metric4/Train&Val': ['Multiline', train_val_sets[3]],
            },
            'Loss': {
                'Loss1/Train&Val': ['Multiline', loss_sets[0]],
                'Loss2/Train&Val': ['Multiline', loss_sets[1]],
                'Loss3/Train&Val': ['Multiline', loss_sets[2]],
                'Loss4/Train&Val': ['Multiline', loss_sets[3]],
            },
            'LearningRate': {'Train': ['Multiline', ['learning_rate']]}
        })
        writer.close()

        if self.plot_to_disk:
            logger.error('plot_and_save not implemented yet to handle the new tweaks')
            # TODO: implement a proper plot_and_save for to the tweaks
            # self.plot_and_save(data_logger, step_divider)

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

        _, metrics, _ = self.validation(dataloader=self.test_loader, testing=True, **kwargs)

        for idx, module in enumerate(self.module.module_names):
            for k, v in metrics[idx].items():
                logger.info(f'Test output {idx+1} - {module} {k}: {v.item():.6f}')
