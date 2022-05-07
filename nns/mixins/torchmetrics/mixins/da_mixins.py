# -*- coding: utf-8 -*-
""" nns/mixins/torchmetrics/mixins/da_mixins """

from statistics import mean

from logzero import logger
from torchmetrics import MetricCollection

from nns.mixins.torchmetrics.mixins.base import TorchMetricsBaseMixin


__all__ = ['DATorchMetricsMixin']


class DATorchMetricsMixin(TorchMetricsBaseMixin):
    """
    Provides methods to initialize the DAModelMGR and handle the MetricCollection objects

    Usage:
       MyModelMGR(DATorchMetricsMixin):
           def __init__(self, **kwargs):
               self._TorchMetricsBaseMixin__init(**kwargs)

    """
    train_prefix1 = 'train1_'
    valid_prefix1 = 'val1_'
    train_prefix2 = 'train2_'
    valid_prefix2 = 'val2_'

    def _init_subdataset_metrics(self, metrics_tmp: list):
        """
        Initializes the subdataset metrics

        Note: overwrite this method as necessary

        Kwargs:
            metrics_tmp: list of MetricItem instances
        """
        assert isinstance(metrics_tmp, list), type(metrics_tmp)

        metrics_tmp = MetricCollection(metrics_tmp)
        self.train_metrics1 = metrics_tmp.clone(prefix=self.train_prefix1)
        self.valid_metrics1 = metrics_tmp.clone(prefix=self.valid_prefix1)
        self.train_metrics2 = metrics_tmp.clone(prefix=self.train_prefix2)
        self.valid_metrics2 = metrics_tmp.clone(prefix=self.valid_prefix2)

    def print_epoch_summary(self, epoch: int, data_logger: dict):
        """
        Prints and epoch summary

        Kwargs:
            epoch        <int>: Current zero-based epoch
            data_logger <dict>: Dictionary of training logs
        """
        assert isinstance(epoch, int), type(epoch)
        assert isinstance(data_logger, dict), type(data_logger)

        mtrain_loss = mean([data_logger["epoch_train_losses1"][epoch],
                           data_logger["epoch_train_losses2"][epoch]])
        mval_loss = mean([data_logger["epoch_val_losses1"][epoch],
                          data_logger["epoch_val_losses2"][epoch]])

        text = f'Epoch {epoch+1}\n' + \
            f'mean train loss: {mtrain_loss:.6f} \t\t' + \
            f'mean val loss: {mval_loss:.6f}\n'

        logger.info(
            text + 'Model1:\n' +
            self.metrics_to_str(self.prepare_to_save(data_logger['epoch_train_metrics1'][epoch]),
                                self.prepare_to_save(data_logger['epoch_val_metrics1'][epoch])) +
            'Model2:\n' +
            self.metrics_to_str(self.prepare_to_save(data_logger['epoch_train_metrics2'][epoch]),
                                self.prepare_to_save(data_logger['epoch_val_metrics2'][epoch]))
        )
