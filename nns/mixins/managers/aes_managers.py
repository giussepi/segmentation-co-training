# -*- coding: utf-8 -*-
""" nns/mixins/managers/aes_managers """

import os
from unittest.mock import MagicMock

import numpy as np
import torch
from gtorch_utils.nns.managers.callbacks import Checkpoint, EarlyStopping, \
    MetricEvaluator, MaskPlotter
from gtorch_utils.nns.mixins.constants import LrShedulerTrack
from gtorch_utils.nns.mixins.managers.base import BaseModelMGR
from gutils.folders import clean_create_folder
from logzero import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR


__all__ = ['AEsModelMGRMixin']


class AEsModelMGRMixin(BaseModelMGR):
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
                                          gtorch_utils.nns.managers.callbacks.plotters.masks import MaskPlotter
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
                    epoch_train_loss += sum(v.item() for v in batch_train_loss.values())
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
