# -*- coding: utf-8 -*-
""" nns/mixins/torchmetrics/mixins/da_mixins """

from statistics import mean

import torch
from logzero import logger
from torchmetrics import MetricCollection

from nns.mixins.torchmetrics.mixins.base import TorchMetricsBaseMixin


__all__ = ['ModularUNet4PlusTorchMetricsMixin']


class ModularUNet4PlusTorchMetricsMixin(TorchMetricsBaseMixin):
    """
    Provides methods to initialize the ModelMGR and handle the MetricCollection objects for
    ModularUNet4Plus

    Usage:
       MyModelMGR(ModularUNet4PlusTorchMetricsMixin):
           def __init__(self, **kwargs):
               self._TorchMetricsBaseMixin__init(**kwargs)

    """

    def _init_subdataset_metrics(self, metrics_tmp: list):
        """
        Initializes the subdataset metrics

        Note: overwrite this method as necessary

        Kwargs:
            metrics_tmp: list of MetricItem instances
        """
        assert isinstance(metrics_tmp, list), type(metrics_tmp)

        for idx, module in enumerate(self.module.module_names, start=1):
            setattr(self, f'train_prefix{idx}', f'train{module}_')
            setattr(self, f'valid_prefix{idx}', f'val{module}_')

        metrics_tmp = MetricCollection(metrics_tmp)
        self.train_metrics1 = metrics_tmp.clone(prefix=self.train_prefix1)
        self.valid_metrics1 = metrics_tmp.clone(prefix=self.valid_prefix1)
        self.train_metrics2 = metrics_tmp.clone(prefix=self.train_prefix2)
        self.valid_metrics2 = metrics_tmp.clone(prefix=self.valid_prefix2)
        self.train_metrics3 = metrics_tmp.clone(prefix=self.train_prefix3)
        self.valid_metrics3 = metrics_tmp.clone(prefix=self.valid_prefix3)
        self.train_metrics4 = metrics_tmp.clone(prefix=self.train_prefix4)
        self.valid_metrics4 = metrics_tmp.clone(prefix=self.valid_prefix4)

    def print_validation_summary(self, **kwargs):
        """
        Print an summary (this method must be called after performing a validation)

        Kwargs:
            global_step       <int>:
            validation_step   <int>:
            loss     <torch.Tensor>: tensor [1x<number of modules>]
            metrics    <List[dict]>:
            val_loss <List[torch.Tensor]>:
            val_metrics <List[dict]>:
        """
        global_step = kwargs.get('global_step')
        validation_step = kwargs.get('validation_step')
        loss = kwargs.get('loss')
        metrics = kwargs.get('metrics')
        val_loss = kwargs.get('val_loss')
        val_metrics = kwargs.get('val_metrics')

        assert isinstance(global_step, int), type(global_step)
        assert isinstance(validation_step, int), type(validation_step)
        assert isinstance(loss, torch.Tensor), type(loss)
        assert isinstance(metrics, list), type(metrics)
        assert isinstance(val_loss, list), type(val_loss)
        assert isinstance(val_metrics, list), type(val_metrics)

        text = f'Global batch: {global_step} \t\t Validation batch {validation_step}\n'

        module_text = ''

        for idx, _ in enumerate(metrics):
            module_text += f'Train loss {idx+1}: {loss[idx].item():.6f} \t\t' + \
                f'Val loss {idx+1}: {val_loss[idx].item():.6f}\n'
            module_text += self.metrics_to_str(
                self.prepare_to_save(metrics[idx]), self.prepare_to_save(val_metrics[idx]))

        logger.info(text + module_text)

    def print_epoch_summary(self, epoch: int, data_logger: dict):
        """
        Prints and epoch summary

        Kwargs:
            epoch        <int>: Current zero-based epoch
            data_logger <dict>: Dictionary of training logs
        """
        assert isinstance(epoch, int), type(epoch)
        assert isinstance(data_logger, dict), type(data_logger)

        modules = 4

        mtrain_loss = mean([data_logger[f"epoch_train_losses{idx}"][epoch] for idx in range(1, modules+1)])
        mval_loss = mean([data_logger[f"epoch_val_losses{idx}"][epoch] for idx in range(1, modules+1)])

        text = f'Epoch {epoch+1}\n' + \
            f'mean train loss: {mtrain_loss:.6f} \t\t' + \
            f'mean val loss: {mval_loss:.6f}\n'

        for idx in range(1, modules+1):
            text += 'Module {idx}:\n' + \
                self.metrics_to_str(self.prepare_to_save(data_logger[f'epoch_train_metrics{idx}'][epoch]),
                                    self.prepare_to_save(data_logger[f'epoch_val_metrics{idx}'][epoch]))

        logger.info(text)
