# -*- coding: utf-8 -*-
""" nns/mixins/torchmetrics/mixins """

from typing import List, Union, Optional

import torch
from gtorch_utils.segmentation.torchmetrics import DiceCoefficient
from logzero import logger
from torchmetrics import MetricCollection

from nns.mixins.torchmetrics.exceptions import PrepareToSaveError
from nns.utils.metrics import MetricItem


__all__ = ['TorchMetricsMixin']


class TorchMetricsMixin:
    """
    Provides methods to initialize the ModelMGR and handle the MetricCollection object

    Usage:
       MyModelMGR(TorchMetricsMixin):
           def __init__(self, **kwargs):
               self._TorchMetricsMixin__init(**kwargs)
               ...
    """

    train_prefix = 'train_'
    valid_prefix = 'val_'

    def __init(self, **kwargs):
        """
        Validates the keyword arguments and initializes object instance

        Kwargs:
            metrics <dict>: List of MetricItems to be used by the manager
                            Default [MetricItem(DiceCoefficient(), main=True),]
        """
        self.metrics = kwargs.get('metrics', [MetricItem(DiceCoefficient(), main=True), ])

        assert isinstance(self.metrics, list), type(self.metrics)

        metrics_tmp = []
        self.main_metrics = []

        for metric_item in self.metrics:
            assert isinstance(metric_item, MetricItem), type(metric_item)

            metrics_tmp.append(metric_item.metric)

            if metric_item.main:
                self.main_metrics.append(metric_item.metric.__class__.__name__)

        assert len(self.main_metrics) > 0, 'At least one MetricItem must have main=True'

        metrics_tmp = MetricCollection(metrics_tmp)
        self.train_metrics = metrics_tmp.clone(prefix=self.train_prefix)
        self.valid_metrics = metrics_tmp.clone(prefix=self.valid_prefix)

    @property
    def main_metrics_str(self) -> str:
        """
        Returns the contatenation of the main metrics names

        Returns:
            metric_names_concatenated <str>
        """
        return '+'.join(self.main_metrics)

    def get_mean_main_metrics(self, metrics: dict) -> float:
        """
        Returns the mean of all main metrics from the metrics dictionary provided

        Kwargs:
            metrics <dict>: Dictionary obtained by calling MyMetricCollection(preds, target) or
                            MyMetricCollection.compute()

        Returns:
            combined_metric <float>
        """
        assert isinstance(metrics, dict), type(metrics)
        assert len(metrics) > 0, 'metrics cannot be an empty dictionary'

        combined_metric = 0.

        prefix = [*metrics.keys()][0].split('_')[0] + '_'

        if prefix not in [self.train_prefix, self.valid_prefix]:
            prefix = ''

        for metric in self.main_metrics:
            combined_metric += metrics[prefix + metric]

        return combined_metric.item() / len(self.main_metrics)

    @staticmethod
    def sum_metrics(metrics1: dict, metrics2: dict, /) -> dict:
        """
        Returns the element wise summation of the metrics dictionaries provided. They should have been
        generated using a MetricCollection

        Kwargs:
            metrics1 <dict>: Dictionary obtained by calling MyMetricCollection(preds, target) or
                             MyMetricCollection.compute()
            metrics2 <dict>: Dictionary obtained by calling MyMetricCollection(preds, target) or
                             MyMetricCollection.compute()
        """
        assert isinstance(metrics1, dict), type(metrics1)
        assert isinstance(metrics2, dict), type(metrics2)

        return {k: metrics1.get(k, torch.tensor(0.)) + metrics2.get(k, torch.tensor(0.))
                for k in set(metrics1) | set(metrics2)}

    @staticmethod
    def metrics_to_str(metrics1: dict, metrics2: Optional[dict] = None) -> str:
        """
        Returns a ready-to-print human-readable representation of the provided metrics.
        Works with one or two metrics dictionaries. If two metric dicts are provided
        the they will be nicely formatted as training vs validation metrics

        Kwargs:
            metrics1 <dict>: Dictionary obtained by calling MyMetricCollection(preds, target) or
                             MyMetricCollection.compute()
            metrics2 <dict>: [OPTIONAL] Dictionary obtained by calling MyMetricCollection(preds, target)
                             or MyMetricCollection.compute()

        Returns:
            metrics_stx <str>
        """
        assert isinstance(metrics1, dict), type(metrics1)

        text = ''

        if metrics2 is not None:
            assert isinstance(metrics2, dict), type(metrics2)

            for (train_k, train_v), (val_k, val_v) in zip(metrics1.items(), metrics2.items()):
                text += f'{train_k}: {train_v:.6f} \t\t {val_k}: {val_v:.6f}\n'
        else:
            for key, value in metrics1.items():
                text += f'{key}: {value:.6f}\n'

        return text

    @staticmethod
    def _prepare_to_save_dict(metrics: dict) -> dict:
        """
        Return a copy of the dictionary without Tensors

        Kwargs:
            metrics <dict>: Dictionary obtained by calling MyMetricCollection(preds, target) or
                            MyMetricCollection.compute()

        Returns:
            ready_to_save_metrics <dict>
        """
        assert isinstance(metrics, dict), type(metrics)

        try:
            return {k: v.item() for k, v in metrics.items()}
        except AttributeError:
            return metrics

    def _prepare_to_save_list_dict(self, metrics_list: List[dict]) -> List[dict]:
        """
        Returns a copy of the list of dictionaries without Tensors

        Kwargs:
            metrics_list <list>: List of dictionaries obtained by calling
                                 MyMetricCollection(preds, target) or MyMetricCollection.compute()

        Returns:
            cleaned_metrics_list <List[dict]>
        """
        assert isinstance(metrics_list, list), type(metrics_list)

        return [self._prepare_to_save_dict(metrics) for metrics in metrics_list]

    @staticmethod
    def _prepare_to_save_list(mylist: List[torch.Tensor]) -> List[float]:
        """
        Returns a copy of the List[torch.Tensor] but containing only float values

        Returns:
            cleaned_list <List[float]>
        """
        assert isinstance(mylist, list), type(mylist)

        try:
            return [element.item() for element in mylist]
        except AttributeError:
            return mylist

    def prepare_to_save(
            self, data: Union[dict, List[dict], List[torch.Tensor]]) -> Union[dict, List[dict], List[float]]:
        """
        Functor method to get rid of tensors from the provided dada

        Kwargs:
            data <dict | List[dict] | List[torch.Tensor]>: data to be cleaned of tensors

        Returns:
            cleaned_data <dict | List[dict] | List[float]>
        """
        if isinstance(data, dict):
            return self._prepare_to_save_dict(data)

        if isinstance(data, list):
            if isinstance(data[0], dict):
                return self._prepare_to_save_list_dict(data)

            if isinstance(data[0], torch.Tensor):
                return self._prepare_to_save_list(data)

        raise PrepareToSaveError()

    def print_validation_summary(self, **kwargs):
        """
        Print an summary (this method must be called after performing a validation)

        Kwargs:
            global_step       <int>:
            validation_step   <int>:
            loss     <torch.Tensor>:
            metrics          <dict>:
            val_loss <torch.Tensor>:
            val_metrics      <dict>:
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
        assert isinstance(metrics, dict), type(metrics)
        assert isinstance(val_loss, torch.Tensor), type(val_loss)
        assert isinstance(val_metrics, dict), type(val_metrics)

        text = f'Global batch: {global_step} \t\t Validation batch {validation_step}\n' + \
            f'Train loss: {loss.item():.6f} \t\t Val loss: {val_loss.item():.6f}\n'

        logger.info(
            text + self.metrics_to_str(self.prepare_to_save(metrics), self.prepare_to_save(val_metrics)))

    def print_epoch_summary(self, epoch: int, data_logger: dict):
        """
        Prints and epoch summary

        Kwargs:
            epoch        <int>: Current zero-based epoch
            data_logger <dict>: Dictionary of training logs
        """
        assert isinstance(epoch, int), type(epoch)
        assert isinstance(data_logger, dict), type(data_logger)

        text = f'Epoch {epoch+1}\n' + \
            f'train loss: {data_logger["epoch_train_losses"][epoch]:.6f} \t\t' + \
            f'val loss: {data_logger["epoch_val_losses"][epoch]:.6f}\n'

        logger.info(
            text + self.metrics_to_str(self.prepare_to_save(data_logger['epoch_train_metrics'][epoch]),
                                       self.prepare_to_save(data_logger['epoch_val_metrics'][epoch]))
        )

    def get_best_combined_main_metrics(self, metrics_list: List[dict]) -> float:
        """
        Calculates and returns the best combined main metrics

        Kwargs:
            metrics_list <list>: List of metrics <dict>

        Returns:
            best_combined_main_metrics <float>
        """
        assert isinstance(metrics_list, list), type(metrics_list)

        return max([self.get_mean_main_metrics(metrics) for metrics in metrics_list]).item()
