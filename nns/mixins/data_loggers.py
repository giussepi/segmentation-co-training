# -*- coding: utf-8 -*-
""" nns/mixins/loggers """

import os

import numpy as np
from logzero import logger
from tabulate import tabulate
from tqdm import tqdm

from nns.callbacks.plotters.training import TrainingPlotter


class DataLoggerMixin:
    """
    Provides methods to plot and save data from data_logger

    Usage:
        class SomeClass(DataLoggerMixin):
            ...
    """

    def plot_and_save(self, data_logger, step_divider):
        """
        Plots the data from data_logger and save it to disk

        Args:
            data_logger <dict>: Dictionary of training logs
            step_divider <int>: step divider used during training
        """
        assert isinstance(data_logger, dict), type(data_logger)
        assert isinstance(step_divider, int), type(step_divider)

        logger.info("Plotting and saving training data")

        with tqdm(total=4, unit='img') as pbar:
            TrainingPlotter(
                train_loss=data_logger['train_loss'],
                train_metric=data_logger['train_metric'],
                val_loss=data_logger['val_loss'],
                val_metric=data_logger['val_metric'],
                lr=data_logger['lr'],
                clip_interval=(0, 1),
            )(
                save=True, show=False, xlabel=f'Sets of {step_divider} images',
                lm_saving_path=os.path.join(self.plot_dir, 'losses_metrics.png'),
                lr_saving_path=os.path.join(self.plot_dir, 'learning_rate.png')
            )
            pbar.update(2)

            TrainingPlotter(
                train_loss=data_logger['epoch_train_losses'],
                train_metric=data_logger['epoch_train_metrics'],
                val_loss=data_logger['epoch_val_losses'],
                val_metric=data_logger['epoch_val_metrics'],
                lr=data_logger['epoch_lr'],
                clip_interval=(0, 1),
            )(
                save=True, show=False,
                lm_saving_path=os.path.join(self.plot_dir, 'epoch_losses_metrics.png'),
                lr_saving_path=os.path.join(self.plot_dir, 'epoch_learning_rate.png')
            )
            pbar.update(2)

    def print_data_logger_summary(self, tablefmt='orgtbl'):
        """
        Prints a summary of the data_logger for the provided ini_checkpoint.
        Always use it with the last checkpoint saved to include all the logs when generating
        the summary table

        Args:
            tablefmt      <str>: format to be used. See https://pypi.org/project/tabulate/
                                 Default 'orgtbl'
        """
        assert isinstance(tablefmt, str), type(tablefmt)

        # Loading the provided checkpoint or the best model obtained during training
        if self.ini_checkpoint:
            _, data_logger = self.load_checkpoint(self.optimizer(self.model.parameters(), **self.optimizer_kwargs))
        else:
            self.load()

        data = [["key", "Validation", "corresponding training value"]]
        max_idx = np.argmax(data_logger['val_metric'])

        data.append(["Best metric",
                     f"{data_logger['val_metric'][max_idx]:.4f}",
                     f"{data_logger['train_metric'][max_idx]:.4f}"])
        min_idx = np.argmin(data_logger['val_loss'])
        data.append(["Min loss",
                     f"{data_logger['val_loss'][min_idx]:.4f}",
                     f"{data_logger['train_loss'][min_idx]:.4f}"])
        data.append(["Max LR", "", f"{max(data_logger['lr']):.0e}"])
        data.append(["Min LR", "", f"{min(data_logger['lr']):.0e}"])

        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
