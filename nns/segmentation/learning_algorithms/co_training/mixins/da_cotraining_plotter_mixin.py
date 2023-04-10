# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/mixins/da_cotraining_plotter_mixin """

import os
from typing import Dict, Union, List

import numpy as np
import torch
from gtorch_utils.nns.managers.callbacks import TrainingPlotter
from tabulate import tabulate

from nns.segmentation.learning_algorithms.co_training.mixins.base_cotraining_plotter_mixin import \
    BaseCotrainingPlotterMixin


__all__ = ['DACotrainingPlotterMixin']


class DACotrainingPlotterMixin(BaseCotrainingPlotterMixin):
    """
    Contains methods to plot data from class CoTraining

    Usage:
        class DACoTraining(DACotrainingPlotterMixin, ...):
            ...
    """

    def plot_and_save(self, checkpoint: str, save: bool = False, dpi: str = 'figure', show: bool = True):
        """
        Plosts and saves (optionally) the co-training data_logger from the 'checkpoint'

        Kwargs:
            checkpoint <str>: path to the CoTraining checkpoint
            save           <bool>: Whether or not save to disk. Default False
            dpi <float, 'figure'>: The resolution in dots per inch.
                                   If 'figure', use the figure's dpi value. For high quality images
                                   set it to 300.
                                   Default 'figure'
            show           <bool>: Where or not display the image. Default True
        """
        assert os.path.isfile(checkpoint), f'{checkpoint} does not exist.'
        assert isinstance(save, bool), type(save)
        assert isinstance(dpi, (float, str)), type(dpi)
        assert isinstance(show, bool), type(show)

        data = torch.load(checkpoint)['data_logger']

        # plotting metrics and losses
        for idx, model_cls in enumerate([self.model_mgr_kwargs['model_kwargs']['model1_cls'],
                                         self.model_mgr_kwargs['model_kwargs']['model2_cls']]):
            data_logger_val_models_metrics = self.get_separate_metrics(
                data['val_models_metrics'], model_idx=idx)
            data_logger_train_models_metrics = self.get_separate_metrics(
                data['train_models_metrics'], model_idx=idx)

            for metric in data_logger_val_models_metrics:
                metric_name = metric.split('_')[-1]
                xticks_labels = np.arange(1, len(data_logger_val_models_metrics[metric])+1)
                TrainingPlotter(
                    train_metric=data_logger_train_models_metrics[metric.replace('valid', 'train')],
                    val_metric=data_logger_val_models_metrics[metric]
                )(
                    lm_title=f'Model {idx+1} ({model_cls.__name__}): Metrics',
                    xlabel='Co-training epochs',
                    lm_ylabel=metric_name,
                    train_metric_label='train',
                    val_metric_label='val',
                    lm_xticks_labels=xticks_labels,
                    lm_legend_kwargs=dict(shadow=True, fontsize=8, loc='best'),
                    lm_saving_path=os.path.join(
                        self.plots_saving_path, f'model_{idx+1}_{metric_name}.png'),
                    save=save,
                    dpi=dpi,
                    show=show,
                )

            TrainingPlotter(
                train_loss=torch.as_tensor(data['train_models_losses'])[:, idx].detach().cpu().tolist(),
                val_loss=torch.as_tensor(data['val_models_losses'])[:, idx].detach().cpu().tolist(),
            )(
                lm_title=f'Model {idx+1} ({model_cls.__name__}): Losses',
                xlabel='Co-training epochs',
                lm_ylabel='Loss',
                train_loss_label='train',
                val_loss_label='val',
                lm_xticks_labels=xticks_labels,
                lm_legend_kwargs=dict(shadow=True, fontsize=8, loc='best'),
                lm_saving_path=os.path.join(
                    self.plots_saving_path, f'model_{idx+1}_losses.png'),
                save=save,
                dpi=dpi,
                show=show,
            )

        # plotting new masks metrics and combined preds metrics
        data_logger_train_new_masks_metric = self.get_separate_metrics(data['train_new_masks_metric'])
        data_logger_train_combined_preds_metric = self.get_separate_metrics(
            data['train_combined_preds_metric'])
        data_logger_val_combined_preds_metric = self.get_separate_metrics(data['val_combined_preds_metric'])

        for metric1, metric2, metric3 in zip(data_logger_train_new_masks_metric,
                                             data_logger_train_combined_preds_metric,
                                             data_logger_val_combined_preds_metric):
            metric_name = metric1.split('_')[-1]
            TrainingPlotter(
                train_loss=data_logger_train_new_masks_metric[metric1],
                train_metric=data_logger_train_combined_preds_metric[metric2],
                val_loss=data['val_new_masks_metric'],
                val_metric=data_logger_val_combined_preds_metric[metric3]
            )(
                lm_title='New masks and combined masks metrics',
                xlabel='Co-training iterations',
                lm_ylabel=metric_name,
                lm_xticks_labels=np.arange(1, len(data_logger_train_new_masks_metric[metric1])+1),
                lm_legend_kwargs=dict(shadow=True, fontsize=8, loc='best'),
                lm_saving_path=os.path.join(
                    self.plots_saving_path, f'new_masks_combined_preds_{metric_name}.png'),
                train_loss_label='train new masks',
                val_loss_label='val new masks',
                train_metric_label='train combined preds',
                val_metric_label='val combined preds',
                save=save,
                dpi=dpi,
                show=show
            )

    def print_data_logger_summary(self, checkpoint: str, tablefmt: str = 'orgtbl'):
        """
        Prints a summary of the data_logger for the provided ini_checkpoint.
        Always use it with the last checkpoint saved to include all the logs when generating
        the summary table

        Kwargs:
            checkpoint <str>: path to the CoTraining checkpoint
            tablefmt   <str>: format to be used. See https://pypi.org/project/tabulate/
                              Default 'orgtbl'
        """
        assert os.path.isfile(checkpoint), f'{checkpoint} does not exist.'
        assert isinstance(tablefmt, str), type(tablefmt)

        data_logger = torch.load(checkpoint)['data_logger']

        # plotting metrics and losses
        for idx, model_cls in enumerate([self.model_mgr_kwargs['model_kwargs']['model1_cls'],
                                         self.model_mgr_kwargs['model_kwargs']['model2_cls']]):
            data = [["key", "Validation", "corresponding training value", "Epoch"]]

            data_logger_val_models_metrics = self.get_separate_metrics(
                data_logger['val_models_metrics'], model_idx=idx, np_array=True)
            data_logger_train_models_metrics = self.get_separate_metrics(
                data_logger['train_models_metrics'], model_idx=idx, np_array=True)
            data_logger['val_models_losses'] = np.array(data_logger['val_models_losses'])
            data_logger['train_models_losses'] = np.array(data_logger['train_models_losses'])

            for metric in data_logger_val_models_metrics:
                max_idx = np.argmax(data_logger_val_models_metrics[metric])
                data.append([
                    f"Best_{metric.split('_')[-1]}",
                    f"{data_logger_val_models_metrics[metric][max_idx]:.4f}",
                    f"{data_logger_train_models_metrics[metric.replace('valid', 'train')][max_idx]:.4f}",
                    f"{max_idx+1}"
                ])

            min_idx = np.argmin(data_logger['val_models_losses'][:, idx])
            data.append(["Min loss",
                         f"{data_logger['val_models_losses'][min_idx, idx]:.4f}",
                         f"{data_logger['train_models_losses'][min_idx, idx]:.4f}",
                         f"{min_idx+1}"])

            print(f'MODEL {idx+1} ({model_cls.__name__}):')
            print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
            print('\n')

        # plotting new masks metrics and combined preds metrics
        data = [["key", "training value", "Epoch"]]

        data_logger_train_new_masks_metric = self.get_separate_metrics(
            data_logger['train_new_masks_metric'], np_array=True)

        for metric in data_logger_train_new_masks_metric:
            max_idx = np.argmax(data_logger_train_new_masks_metric[metric])
            data.append([
                f"Best_{metric}",
                f"{data_logger_train_new_masks_metric[metric][max_idx]:.4f}",
                f"{max_idx+1}"
            ])

        print('New masks')
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
        print('\n')

        data_logger_val_combined_preds_metric = self.get_separate_metrics(
            data_logger['val_combined_preds_metric'], np_array=True)
        data_logger_train_combined_preds_metric = self.get_separate_metrics(
            data_logger['train_combined_preds_metric'], np_array=True)
        data = [["key", "Validation", "corresponding training value", "Epoch"]]

        for metric in data_logger_val_combined_preds_metric:
            max_idx = np.argmax(data_logger_val_combined_preds_metric[metric])
            data.append([
                f"Best_{metric.split('_')[-1]}",
                f"{data_logger_val_combined_preds_metric[metric][max_idx]:.4f}",
                f"{data_logger_train_combined_preds_metric[metric.replace('valid', 'train')][max_idx]:.4f}",
                f"{max_idx+1}"
            ])

        print('Combined predictions metrics')
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
        print('\n')

    @staticmethod
    def print_all_metrics(
            data_logger_metric: Dict[str, Union[list, np.ndarray]], title: str, /,
            *, tablefmt: str = 'orgtbl'
    ):
        """
        Prints a table with all metrics per epoch

        Kwargs:
            data_logger_metric <Dict[str, Union[list, np.ndarray]]>: Dictionary containing the
                               metric names a keys and a list or np.array of its values per
                               cotraining epoch
            title       <str>: Table title
            tablefmt    <str>: format to be used. See https://pypi.org/project/tabulate/
                               Default 'orgtbl'
        """
        assert isinstance(data_logger_metric, dict), \
            type(data_logger_metric)
        assert isinstance(title, str), type(title)
        assert isinstance(tablefmt, str), type(tablefmt)

        data = [[] for _ in range([*data_logger_metric.values()][0].shape[0])]
        data.insert(0, [])

        for metric in data_logger_metric:
            data[0].append(metric.split('_')[-1])

            for score_idx, score in enumerate(data_logger_metric[metric], start=1):
                data[score_idx].append(f"{score:.4f}")

        print(f'{title}')
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
        print('\n')

    def print_all_losses(
            self, data_logger_models_losses: List[list], title: str, /, *, tablefmt: str = 'orgtbl'):
        """
        Prints a table with all model losses per epoch

        Kwargs:
            data_logger_models_losses <List[list]>: List containing lists of models losses per epoch.
                               e.g. [[model1_score1, model2_score1], [model1_score2, model2_score2], ...]
            title       <str>: Table title
            tablefmt    <str>: format to be used. See https://pypi.org/project/tabulate/
                               Default 'orgtbl'
        """
        assert isinstance(data_logger_models_losses, list), type(data_logger_models_losses)
        assert isinstance(title, str), type(title)
        assert isinstance(tablefmt, str), type(tablefmt)

        data = [[f'{model_cls.__name__}' for model_cls in [
            self.model_mgr_kwargs['model_kwargs']['model1_cls'],
            self.model_mgr_kwargs['model_kwargs']['model2_cls']
        ]]]

        data.extend(data_logger_models_losses)
        print(title)
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
        print('\n')

    def print_data_logger_details(self, checkpoint: str, /, *, tablefmt: str = 'orgtbl'):
        """
        Prints a summary of the data_logger for the provided ini_checkpoint.
        Always use it with the last checkpoint saved to include all the logs when generating
        the summary table

        Kwargs:
            checkpoint <str>: path to the CoTraining checkpoint
            tablefmt   <str>: format to be used. See https://pypi.org/project/tabulate/
                              Default 'orgtbl'
        """
        assert os.path.isfile(checkpoint), f'{checkpoint} does not exist.'
        assert isinstance(tablefmt, str), type(tablefmt)

        data_logger = torch.load(checkpoint)['data_logger']

        # printing metrics per model
        for idx, model_cls in enumerate([self.model_mgr_kwargs['model_kwargs']['model1_cls'],
                                         self.model_mgr_kwargs['model_kwargs']['model2_cls']]):
            self.print_all_metrics(
                self.get_separate_metrics(data_logger['train_models_metrics'], model_idx=idx, np_array=True),
                f'MODEL {idx+1} ({model_cls.__name__}) [TRAIN]',
                tablefmt=tablefmt
            )
            self.print_all_metrics(
                self.get_separate_metrics(data_logger['val_models_metrics'], model_idx=idx, np_array=True),
                f'MODEL {idx+1} ({model_cls.__name__}) [VALIDATION]',
                tablefmt=tablefmt
            )

        # printing losses per model
        self.print_all_losses(data_logger['train_models_losses'], "TRAIN LOSS", tablefmt=tablefmt)
        self.print_all_losses(data_logger['val_models_losses'], "VALIDATION LOSS", tablefmt=tablefmt)

        # printing new masks metrics and combined preds metrics
        self.print_all_metrics(
            self.get_separate_metrics(data_logger['train_new_masks_metric'], np_array=True),
            'TRAIN new masks',
            tablefmt=tablefmt
        )
        self.print_all_metrics(
            self.get_separate_metrics(data_logger['train_combined_preds_metric'], np_array=True),
            "TRAIN combined predictions",
            tablefmt=tablefmt
        )
        self.print_all_metrics(
            self.get_separate_metrics(data_logger['val_combined_preds_metric'], np_array=True),
            'VALIDATION combined predictions',
            tablefmt=tablefmt
        )
