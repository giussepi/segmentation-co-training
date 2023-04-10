# -*- coding: utf-8 -*-
""" nns/mixins/managers/adsv_managers """

from unittest.mock import MagicMock

import numpy as np
import torch
from gtorch_utils.nns.managers.callbacks import Checkpoint, EarlyStopping, MetricEvaluator
from gtorch_utils.nns.mixins.constants import LrShedulerTrack
from logzero import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nns.mixins.managers.base import BaseModelMGR
from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR


__all__ = ['ADSVModelMGRMixin']


class ADSVModelMGRMixin(BaseModelMGR):
    """
    General Alternating Deep Supervision segmentation model manager

    Usage:
        class MyModelMGR(ADSVModelMGRMixin):
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
            self.memory_printer(epoch)

            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img',
                      disable=DISABLE_PROGRESS_BAR) as pbar:

                for batch in self.train_loader:

                    # if (global_step + 1) % step_divider == 0 or (global_step + 1) % train_batches == 0:
                    #     # making sure to update the all the layers before performing and evaluation
                    #     self.module.fwd_counter = 3

                    # passing 4 times each batch to make sure that all the DSVs have been applied
                    for i in range(4):
                        pred, true_masks, imgs, batch_train_loss, metrics, labels, label_names = \
                            self.training_step(batch)
                        # tracking only the last loss
                        if i == 3:
                            epoch_train_loss += batch_train_loss.item()
                        optimizer.zero_grad()

                        if self.cuda:
                            scaler.scale(batch_train_loss).backward(retain_graph=True)
                            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            batch_train_loss.backward(retain_graph=True)
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
