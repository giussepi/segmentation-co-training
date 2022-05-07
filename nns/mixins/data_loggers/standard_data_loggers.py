# -*- coding: utf-8 -*-
""" nns/mixins/data_loggers/standard_data_loggers """

import os

import numpy as np
from logzero import logger
from tabulate import tabulate
from tqdm import tqdm

from nns.callbacks.plotters.training import TrainingPlotter
from nns.mixins.exceptions import IniCheckpintError
from nns.mixins.torchmetrics.mixins import TorchMetricsMixin


class DataLoggerMixin(TorchMetricsMixin):
    """
    Provides methods to plot and save data from data_logger

    Usage:
        class SomeClass(DataLoggerMixin):
            ...
    """

    def plot_and_save(self, data_logger: dict, step_divider: int, dpi: str = 'figure'):
        """
        Plots the data from data_logger and save it to disk

        Args:
            data_logger <dict>: Dictionary of training logs
            step_divider <int>: step divider used during training
            dpi <Union[float, str]>: The resolution in dots per inch.
                                   If 'figure', use the figure's dpi value. For high quality images
                                   set it to 300.
                                   Default 'figure'
        """
        assert isinstance(data_logger, dict), type(data_logger)
        assert isinstance(step_divider, int), type(step_divider)
        assert isinstance(dpi, (float, str)), type(dpi)

        logger.info("Plotting and saving training data")
        data_logger_train_metrics = self.get_separate_metrics(data_logger['train_metric'])
        data_logger_val_metrics = self.get_separate_metrics(data_logger['val_metric'])
        data_logger_epoch_train_metrics = self.get_separate_metrics(data_logger['epoch_train_metrics'])
        data_logger_epoch_val_metrics = self.get_separate_metrics(data_logger['epoch_val_metrics'])

        with tqdm(total=4, unit='img') as pbar:
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
                    lm_saving_path=os.path.join(self.plot_dir, f'{metric_name}.png'),
                    dpi=dpi
                )
            pbar.update(1)

            TrainingPlotter(
                train_loss=data_logger['train_loss'],
                val_loss=data_logger['val_loss'],
                lr=data_logger['lr'],
                clip_interval=(0, 1),
            )(
                save=True, show=False, xlabel=f'Sets of {step_divider} images',
                train_loss_label='train',
                val_loss_label='val',
                lm_ylabel='Loss',
                lm_saving_path=os.path.join(self.plot_dir, 'losses_metrics.png'),
                lr_saving_path=os.path.join(self.plot_dir, 'learning_rate.png')
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
                    lm_saving_path=os.path.join(self.plot_dir, f'epoch_{metric}.png'),
                    dpi=dpi
                )
            pbar.update(1)

            xticks_labels = np.arange(1, len(data_logger['epoch_train_losses'])+1)
            TrainingPlotter(
                train_loss=data_logger['epoch_train_losses'],
                val_loss=data_logger['epoch_val_losses'],
                lr=data_logger['epoch_lr'],
                clip_interval=(0, 1),
            )(
                save=True, show=False,
                train_metric_label='train',
                val_metric_label='val',
                lm_ylabel='Loss',
                lm_xticks_labels=xticks_labels,
                lr_xticks_labels=xticks_labels,
                lm_saving_path=os.path.join(self.plot_dir, 'epoch_losses.png'),
                lr_saving_path=os.path.join(self.plot_dir, 'epoch_learning_rate.png')
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

        _, data_logger = self.load_checkpoint(self.optimizer(self.model.parameters(), **self.optimizer_kwargs))

        data = [["key", "Validation", "corresponding training value", "valid num"]]
        data_logger_val_metric = self.get_separate_metrics(data_logger['val_metric'], np_array=True)
        data_logger_train_metric = self.get_separate_metrics(data_logger['train_metric'], np_array=True)

        for metric in data_logger_val_metric:
            max_idx = np.argmax(data_logger_val_metric[metric])
            data.append([f"Best {metric.split('_')[-1]}",
                         f"{data_logger_val_metric[metric][max_idx]:.4f}",
                         f"{data_logger_train_metric[metric.replace('val', 'train')][max_idx]:.4f}",
                         f"{max_idx+1}"])

        min_idx = np.argmin(data_logger['val_loss'])
        data.append(["Min loss",
                     f"{data_logger['val_loss'][min_idx]:.4f}",
                     f"{data_logger['train_loss'][min_idx]:.4f}",
                     f"{min_idx+1}"])
        data.append(["Max LR", "", f"{max(data_logger['lr']):.0e}", ""])
        data.append(["Min LR", "", f"{min(data_logger['lr']):.0e}", ""])

        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
