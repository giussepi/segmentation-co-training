# -*- coding: utf-8 -*-
""" nns/mixins/data_loggers/da_data_loggers """

import os
from typing import Optional

import numpy as np
from gtorch_utils.nns.managers.callbacks import TrainingPlotter
from logzero import logger
from tabulate import tabulate
from tqdm import tqdm

from nns.mixins.exceptions import IniCheckpintError
from nns.mixins.torchmetrics.mixins import DATorchMetricsMixin

__all__ = ['DADataLoggerMixin']


class DADataLoggerMixin(DATorchMetricsMixin):
    """
    Provides methods to plot and save data from data_logger

    Usage:
        class SomeClass(DataLoggerMixin):
           def __init__(self, **kwargs):
               self._TorchMetricsBaseMixin__init(**kwargs)
    """

    def plot_and_save(self, step_divider: int, data_logger: Optional[dict] = None, dpi: str = 'figure'):
        """
        Plots the data from data_logger and save it to disk

        Args:
            step_divider      <int>: step divider used during training
            data_logger      <dict>: Dictionary of training logs. Default None
            dpi <Union[float, str]>: The resolution in dots per inch.
                                     If 'figure', use the figure's dpi value. For high quality images
                                     set it to 300. Default 'figure'
        """
        assert isinstance(step_divider, int), type(step_divider)
        if data_logger is not None:
            assert isinstance(data_logger, dict), type(data_logger)
        else:
            _, data_logger = self.load_checkpoint([
                self.optimizer1(self.module.model1.parameters(), **self.optimizer1_kwargs),
                self.optimizer2(self.module.model2.parameters(), **self.optimizer2_kwargs),
            ])
        assert isinstance(dpi, (float, str)), type(dpi)

        logger.info("Plotting and saving training data")
        data_logger_train_metrics1 = self.get_separate_metrics(data_logger['train_metric1'])
        data_logger_val_metrics1 = self.get_separate_metrics(data_logger['val_metric1'])
        data_logger_epoch_train_metrics1 = self.get_separate_metrics(data_logger['epoch_train_metrics1'])
        data_logger_epoch_val_metrics1 = self.get_separate_metrics(data_logger['epoch_val_metrics1'])
        data_logger_train_metrics2 = self.get_separate_metrics(data_logger['train_metric2'])
        data_logger_val_metrics2 = self.get_separate_metrics(data_logger['val_metric2'])
        data_logger_epoch_train_metrics2 = self.get_separate_metrics(data_logger['epoch_train_metrics2'])
        data_logger_epoch_val_metrics2 = self.get_separate_metrics(data_logger['epoch_val_metrics2'])

        data_logger_values = [
            (data_logger_train_metrics1, data_logger_val_metrics1,
             data_logger_epoch_train_metrics1, data_logger_epoch_val_metrics1, '1'),
            (data_logger_train_metrics2, data_logger_val_metrics2,
             data_logger_epoch_train_metrics2, data_logger_epoch_val_metrics2, '2'),
        ]

        with tqdm(total=8, unit='img') as pbar:
            for data_logger_train_metrics, data_logger_val_metrics, data_logger_epoch_train_metrics,\
                    data_logger_epoch_val_metrics, model in data_logger_values:
                for metric in data_logger_val_metrics:
                    metric_name = metric.split('_')[-1]
                    TrainingPlotter(
                        train_metric=data_logger_train_metrics[metric.replace('val', 'train')],
                        val_metric=data_logger_val_metrics[metric],
                        clip_interval=(0, 1),
                    )(
                        save=True, show=False, xlabel=f'Sets of {step_divider} images',
                        train_metric_label='train',
                        val_metric_label='val',
                        lm_ylabel=metric_name,
                        lm_saving_path=os.path.join(self.plot_dir, f'{metric_name}_{model}.png'),
                        dpi=dpi
                    )
                pbar.update(1)

                TrainingPlotter(
                    train_loss=data_logger[f'train_loss{model}'],
                    val_loss=data_logger[f'val_loss{model}'],
                    lr=data_logger[f'lr{model}'],
                    clip_interval=(0, 1),
                )(
                    save=True, show=False, xlabel=f'Sets of {step_divider} images',
                    train_loss_label='train',
                    val_loss_label='val',
                    lm_ylabel='Loss',
                    lm_saving_path=os.path.join(self.plot_dir, 'losses_metrics_{model}.png'),
                    lr_saving_path=os.path.join(self.plot_dir, 'learning_rate_{model}.png')
                )
                pbar.update(1)

                for metric in data_logger_epoch_val_metrics:
                    TrainingPlotter(
                        train_metric=data_logger_epoch_train_metrics[metric.replace('val', 'train')],
                        val_metric=data_logger_epoch_val_metrics[metric],
                        clip_interval=(0, 1),
                    )(
                        save=True, show=False,
                        train_metric_label='train',
                        val_metric_label='val',
                        lm_ylabel=metric_name,
                        lm_saving_path=os.path.join(self.plot_dir, f'epoch_{metric}_{model}.png'),
                        dpi=dpi
                    )
                pbar.update(1)

                xticks_labels = np.arange(1, len(data_logger[f'epoch_train_losses{model}'])+1)
                TrainingPlotter(
                    train_loss=data_logger[f'epoch_train_losses{model}'],
                    val_loss=data_logger[f'epoch_val_losses{model}'],
                    lr=data_logger[f'epoch_lr{model}'],
                    clip_interval=(0, 1),
                )(
                    save=True, show=False,
                    train_metric_label='train',
                    val_metric_label='val',
                    lm_ylabel='Loss',
                    lm_xticks_labels=xticks_labels,
                    lr_xticks_labels=xticks_labels,
                    lm_saving_path=os.path.join(self.plot_dir, 'epoch_losses_{model}.png'),
                    lr_saving_path=os.path.join(self.plot_dir, 'epoch_learning_rate_{model}.png')
                )
                pbar.update(1)

    def print_data_logger_summary(self, tablefmt: str = 'orgtbl'):
        """
        Prints a summary of the data_logger for the provided ini_checkpoint.
        Always use it with the last checkpoint saved to include all the logs when generating
        the summary table

        The ModelMGR instance must be initiazed using the ini_checkpoint parameter

        Args:
            tablefmt      <str>: format to be used. See https://pypi.org/project/tabulate/
                                 Default 'orgtbl'
        """
        assert isinstance(tablefmt, str), type(tablefmt)

        # Loading the provided checkpoint or the best model obtained during training
        if not self.ini_checkpoint:
            raise IniCheckpintError()

        _, data_logger = self.load_checkpoint([
            self.optimizer1(self.module.model1.parameters(), **self.optimizer1_kwargs),
            self.optimizer2(self.module.model2.parameters(), **self.optimizer2_kwargs),
        ])

        data_logger_val_metric1 = self.get_separate_metrics(data_logger['val_metric1'], np_array=True)
        data_logger_val_metric2 = self.get_separate_metrics(data_logger['val_metric2'], np_array=True)
        data_logger_train_metric1 = self.get_separate_metrics(data_logger['train_metric1'], np_array=True)
        data_logger_train_metric2 = self.get_separate_metrics(data_logger['train_metric2'], np_array=True)

        # print data from model 1 #############################################
        data = [["key", "Validation", "corresponding training value", "valid num"]]

        for metric in data_logger_val_metric1:
            max_idx = np.argmax(data_logger_val_metric1[metric])
            data.append([f"Best {metric.split('_')[-1]}",
                         f"{data_logger_val_metric1[metric][max_idx]:.4f}",
                         f"{data_logger_train_metric1[metric.replace('val', 'train')][max_idx]:.4f}",
                         f"{max_idx+1}"])

        min_idx = np.argmin(data_logger['val_loss1'])
        data.append(["Min loss",
                     f"{data_logger['val_loss1'][min_idx]:.4f}",
                     f"{data_logger['train_loss1'][min_idx]:.4f}",
                     f"{min_idx+1}"])
        data.append(["Max LR", "", f"{max(data_logger['lr1']):.0e}", ""])
        data.append(["Min LR", "", f"{min(data_logger['lr1']):.0e}", ""])

        print('MODEL1:\n')
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))

        # print data from model 2 #############################################
        data = [["key", "Validation", "corresponding training value", "valid num"]]

        for metric in data_logger_val_metric2:
            max_idx = np.argmax(data_logger_val_metric2[metric])
            data.append([f"Best {metric.split('_')[-1]}",
                         f"{data_logger_val_metric2[metric][max_idx]:.4f}",
                         f"{data_logger_train_metric2[metric.replace('val', 'train')][max_idx]:.4f}",
                         f"{max_idx+1}"])

        min_idx = np.argmin(data_logger['val_loss2'])
        data.append(["Min loss",
                     f"{data_logger['val_loss2'][min_idx]:.4f}",
                     f"{data_logger['train_loss2'][min_idx]:.4f}",
                     f"{min_idx+1}"])
        data.append(["Max LR", "", f"{max(data_logger['lr2']):.0e}", ""])
        data.append(["Min LR", "", f"{min(data_logger['lr2']):.0e}", ""])

        print('MODEL2:\n')
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
