# -*- coding: utf-8 -*-
""" nns/mixins/managers/modular_managers """

import os
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import torch
from gtorch_utils.nns.managers.callbacks import Checkpoint, EarlyStopping, MetricEvaluator, \
    MaskPlotter
from gtorch_utils.nns.mixins.torchmetrics.mixins import ModularTorchMetricsMixin
from gutils.decorators import timing
from gutils.folders import clean_create_folder
from logzero import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nns.mixins.constants import LrShedulerTrack
from nns.mixins.checkpoints import SeveralOptimizersCheckPointMixin
from nns.mixins.managers.base import BaseModelMGR
from nns.mixins.sanity_checks import WeightsChangingSanityChecksMixin
from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR


__all__ = ['ModularModelMGRMixin']


class ModularModelMGRMixin(
        WeightsChangingSanityChecksMixin, SeveralOptimizersCheckPointMixin,
        ModularTorchMetricsMixin, BaseModelMGR
):
    """
    General segmentation model manager for models composed of several modules trained
    with different optimizers

    Usage:
        class MyModelMGR(ModularModelMGRMixin):
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
            loss<Dict[torch.Tensor]>, metric_scores<List[dict]>, extra_data<dict>
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

        loss = defaultdict(lambda: torch.tensor(0.).to(self.device))
        # the folowing variables will store extra data from the last validation batch
        extra_data = None

        for batch in tqdm(dataloader, total=n_val, desc='Testing round', unit='batch', leave=True,
                          disable=not testing or DISABLE_PROGRESS_BAR):
            loss_, extra_data = self.validation_step(
                batch=batch, testing=testing, plot_to_png=plot_to_png, imgs_counter=imgs_counter,
                mask_plotter=mask_plotter
            )
            for k in loss_.keys():
                loss[k] += loss_[k]
            imgs_counter += self.testval_dataloader_kwargs['batch_size']

        # total metrics over all validation batches
        metrics = []
        for idx in range(1, 5):
            metrics.append(getattr(self, f'valid_metrics{idx}').compute())
            # reset metrics states after each epoch
            getattr(self, f'valid_metrics{idx}').reset()

        if testing and plot_to_png and func_plot_palette is not None:
            func_plot_palette(os.path.join(saving_dir, 'label_palette.png'))

        self.model.train()

        if self.sanity_checks:
            self.enable_sanity_checks()

        for k in loss.keys():
            loss[k] /= n_val

        return loss, metrics, extra_data

    def training(self):
        """
        Trains the model

        NOTE: Override this method only if you need a custom logic for the whole training process
        """
        global_step = 0
        step_divider = self.n_train // (self.intrain_val * self.train_dataloader_kwargs['batch_size'])

        metric_evaluator = MetricEvaluator(self.metric_mode)
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        early_stopped = False
        checkpoint = Checkpoint(self.checkpoint_interval) if self.checkpoint_interval > 0 else None
        start_epoch = 0
        validation_step = 0
        data_logger = dict(
            train_loss1=[], train_loss2=[], train_loss3=[], train_loss4=[],
            train_metric1=[], train_metric2=[], train_metric3=[], train_metric4=[],
            val_loss1=[], val_loss2=[], val_loss3=[], val_loss4=[],
            val_metric1=[], val_metric2=[], val_metric3=[], val_metric4=[],
            lr1=[], lr2=[], lr3=[], lr4=[],
            epoch_train_losses1=[], epoch_train_losses2=[], epoch_train_losses3=[], epoch_train_losses4=[],
            epoch_train_metrics1=[], epoch_train_metrics2=[], epoch_train_metrics3=[],
            epoch_train_metrics4=[],
            epoch_val_losses1=[], epoch_val_losses2=[], epoch_val_losses3=[], epoch_val_losses4=[],
            epoch_val_metrics1=[], epoch_val_metrics2=[], epoch_val_metrics3=[], epoch_val_metrics4=[],
            epoch_lr1=[], epoch_lr2=[], epoch_lr3=[], epoch_lr4=[],
        )
        best_metric = np.NINF
        val_loss_min = np.inf
        train_batches = len(self.train_loader)
        modules = self.module.module_names
        optimizers = tuple([self.optimizer(getattr(self.module, module).parameters(),
                           **self.optimizer_kwargs) for module in modules])

        if self.ini_checkpoint:
            start_epoch, data_logger = self.load_checkpoint(optimizers)
            val_loss_min = min(data_logger['val_loss4'])
            best_metric = self.get_best_combined_main_metrics(data_logger['val_metric4'])
            start_epoch += 1  # increasing it to start at the next epoch

        if self.sanity_checks:
            for optimizer, module_name in zip(optimizers, modules):
                self.add_sanity_checks(optimizer, getattr(self.module, module_name))

        if self.cuda:
            scalers = [torch.cuda.amp.GradScaler(enabled=USE_AMP) for _ in optimizers]

        if self.lr_scheduler is not None:
            schedulers = (self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs) for optimizer in optimizers)
        else:
            schedulers = None

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

                    for idx, key in enumerate(modules):
                        epoch_train_loss[idx] += batch_train_loss[key].item()

                    for idx, optimizer in enumerate(optimizers):
                        optimizer.zero_grad()

                        if self.cuda:
                            scalers[idx].scale(batch_train_loss[modules[idx]]).backward(retain_graph=True)
                            nn.utils.clip_grad_value_(
                                getattr(self.module, f'{modules[idx]}').parameters(), 0.1)
                            scalers[idx].step(optimizer)
                            scalers[idx].update()
                        else:
                            batch_train_loss[modules[idx]].backward(retain_graph=True)
                            nn.utils.clip_grad_value_(
                                getattr(self.module, f'{modules[idx]}').parameters(), 0.1)
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

                        print('\n')
                        for k, v in val_loss.items():
                            print(f'{k}: {v}')

                        # maybe if there's no scheduler then the lr shouldn't be plotted
                        for idx, optimizer in enumerate(optimizers, start=1):
                            writer.add_scalar(
                                f'learning_rate{idx}', optimizer.param_groups[0]['lr'], validation_step)
                            data_logger[f'lr{idx}'].append(optimizer.param_groups[0]['lr'])

                            writer.add_scalar(
                                f'Loss{idx}/train', current_epoch_train_loss[idx-1].item(), validation_step)
                            data_logger[f'train_loss{idx}'].append(current_epoch_train_loss[idx-1].item())

                            for metric_, value_ in metrics[idx-1].items():
                                writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                            data_logger[f'train_metric{idx}'].append(self.prepare_to_save(metrics[idx-1]))
                            writer.add_scalar(
                                f'Loss{idx}/val', val_loss[modules[idx-1]].item(), validation_step)
                            data_logger[f'val_loss{idx}'].append(val_loss[modules[idx-1]].item())

                            for metric_, value_ in val_metrics[idx-1].items():
                                writer.add_scalar(f'{metric_}', value_.item(), validation_step)

                            data_logger[f'val_metric{idx}'].append(self.prepare_to_save(val_metrics[idx-1]))

                        self.print_validation_summary(
                            global_step=global_step, validation_step=validation_step,
                            loss=current_epoch_train_loss,
                            metrics=metrics, val_loss=[val_loss[modules[i]] for i in range(len(modules))],
                            val_metrics=val_metrics
                        )

                        self.validation_post(
                            pred=pred.detach().cpu(), true_masks=true_masks.detach().cpu(), labels=labels,
                            imgs=imgs.detach().cpu(),
                            label_names=label_names, writer=writer, validation_step=validation_step,
                            global_step=global_step, val_extra_data=val_extra_data
                        )

                        # NOTE: the following processes will be led by the last encoder's data
                        # TODO: find out if it's better to apply the early stopping to
                        # val_metric or val_loss

                        val_metric = [0]*len(modules)

                        for idx, metrics in enumerate(val_metrics):
                            val_metric[idx] = self.get_mean_main_metrics(metrics)

                        if self.earlystopping_to_metric:
                            if earlystopping(best_metric, val_metric[-1]):
                                early_stopped = True
                                break
                        elif earlystopping(val_loss[modules[-1]].item(), val_loss_min):
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
                                float(f'{epoch}.{intrain_val_counter}'), optimizers, data_logger,
                                best_chkpt=True
                            )
                            best_metric = val_metric[-1]

                        val_loss_min = min(val_loss_min, val_loss[modules[-1]].item())

                        if self.train_eval_chkpt and checkpoint and checkpoint(epoch):
                            intrain_chkpt_counter += 1
                            self.save_checkpoint(
                                float(f'{epoch}.{intrain_chkpt_counter}'), optimizers, data_logger)

                        if schedulers is not None:
                            # TODO: verify the replacement function is working properly
                            for idx, scheduler in enumerate(schedulers):
                                LrShedulerTrack.step(
                                    self.lr_scheduler_track, scheduler, val_metric[idx],
                                    val_loss[modules[idx]]
                                )

            # computing epoch statistiscs #####################################

            for idx in range(len(modules)):
                data_logger[f'epoch_lr{idx+1}'].append(optimizers[idx].param_groups[0]['lr'])
                data_logger[f'epoch_train_losses{idx+1}'].append(
                    epoch_train_loss[idx].item() / train_batches)
                # total metrics over all training batches
                data_logger[f'epoch_train_metrics{idx+1}'].append(
                    self.prepare_to_save(getattr(self, f'train_metrics{idx+1}').compute()))
                # reset metrics states after each epoch
                getattr(self, f'train_metrics{idx+1}').reset()

            val_loss, val_metric, _ = self.validation(dataloader=self.val_loader)
            for idx in range(len(modules)):
                data_logger[f'epoch_val_losses{idx+1}'].append(val_loss[modules[idx]].item())
                data_logger[f'epoch_val_metrics{idx+1}'].append(self.prepare_to_save(val_metric[idx]))

            self.print_epoch_summary(epoch, data_logger)

            if checkpoint and checkpoint(epoch):
                self.save_checkpoint(epoch, optimizers, data_logger)

            if self.last_checkpoint:
                self.save_checkpoint(float(f'{epoch}'), optimizers, data_logger, last_chkpt=True)

            if early_stopped:
                break

        writer_lr_set = []
        train_val_sets = []
        loss_sets = []
        for idx, _ in enumerate(modules, start=1):
            writer_lr_set.append('learning_rate{idx}')
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
            'LearningRate': {'Train': ['Multiline', writer_lr_set]}
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
            optimizers = tuple([
                self.optimizer(getattr(self.module, module).parameters(), **self.optimizer_kwargs)
                for module in self.module.module_names
            ])
            _, _ = self.load_checkpoint(optimizers)
        else:
            self.load()

        _, metrics, _ = self.validation(dataloader=self.test_loader, testing=True, **kwargs)

        for idx, module in enumerate(self.module.module_names):
            for k, v in metrics[idx].items():
                logger.info(f'Test Module {idx+1} - {module} {k}: {v.item():.6f}')
