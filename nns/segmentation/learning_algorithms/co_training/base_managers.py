# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/base_managers """

import os
import copy
import subprocess
from typing import Tuple, List, Union, OrderedDict

import numpy as np
import torch
from gutils.decorators import timing
from gtorch_utils.nns.managers.callbacks import EarlyStopping
from gtorch_utils.nns.managers.standard import ModelMGR
from gtorch_utils.nns.mixins.subdatasets import SubDatasetsMixin
from gtorch_utils.segmentation import metrics
from logzero import logger
from PIL import Image
from tqdm import tqdm

from nns.segmentation.learning_algorithms.co_training.mixins import CotrainingPlotterMixin
from .settings import DISABLE_PROGRESS_BAR


__all__ = ['BaseCoTraining']


class BaseCoTraining(SubDatasetsMixin, CotrainingPlotterMixin):
    """

    Usage:
        MyCotraining(BaseCoTraining):
            ...

        cot = MyCoTraining(...)
    """

    checkpoint_pattern = 'chkpt_{}.pth.tar'

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            model_mgr_kwargs_list <list, tuple>: list or tuple containing dictionaries to create instances of
                                          ModelMGR. Two dictionaries are required.
            iterations             <int>: Number of iterations to execute the co-training. Default 5
            model_mgr_kwargs_tweaks <list, tuple>: list or tuple containing dictionaries with new values
                                          to update model_mgr_kwargs_list after the first iteration. Default []
                                          It should be used to modify/stabilize the training process when
                                          using warm_start = True. Otherwise, the models could learn too fast
                                          or too slow to make the most of the loaded pre-trained models.
                                          Another option to avoid this issue is setting warm_start = False.
            metrics               <list>: List of MetricItems to be used during co-training
                                          Default [MetricItem(DiceCoefficient(), main=True),]
            earlystopping_kwargs  <dict>: Early stopping configuration.
                                          Default dict(min_delta=1e-3, patience=2)
            warm_start      <dict, None>: Configuration of the warm start. Set it to a dict full of zeroes to
                                          only load the weights (e.g. {'lamda': .0, 'sigma': .0}).
                                          Set it to None to not perform warm start. Default None.
            overall_best_models   <bool>: Set to True to always load the overall best models during the
                                          cotraining operations (when using warm start and right after training
                                          ; thus, the best overall models are used instead of the iteration best
                                          models). If False, the best models in each iteration are employed.
                                          Default False
            dir_checkpoints        <str>: path to the directory where checkpoints will be saved
            thresholds            <dict>: Dictionary containing as keys the strategies to apply, and as values
                                          their thresholds. E.g. dict(agreement=.9, disagreement=(.2, .8)) or
                                          dict(selfcombined=.9). Default dict(disagreement=(.2, .8))
            cot_mask_extension     <str>: co-traning mask extension. Default '.cot.mask.png'
            plots_saving_path      <str>: Path to the folder used to save the plots. Default 'plots'
            strategy_postprocessing <dict>: Dictionary of callables to be applied to the masks generated
                                          by especific estrategies. Default {}.
                                          E.g. dict(disagreement=[ExpandPrediction(), ])
            general_postprocessing <list>: List of callables to be applied to the new cotraining masks before
                                          being saved. Default []. E.g. [ExpandPrediction()]
            postprocessing_threshold <float>: Threshold used to created the combined mask predictions used
                                          during the mask postprocessing (See the get_combined_predictions
                                          method). Default .5
            ###################################################################
            #                         SubDatasetsMixin                        #
            ###################################################################
            dataset    <DatasetTemplate>: Custom dataset class descendant of
                                          gtorch_utils.datasets.segmentation.DatasetTemplate.
                                          See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
            dataset_kwargs        <dict>: keyword arguments for the dataset. Default {}
                                          NOTE: if the dataset is enabled to return the original ground truth
                                          masks ('original_mask') along with the  cotraining masks ('mask');
                                          the former will be merged with the predictions (this helps to
                                          reduce errors as the models improve). Otherwise, the cotraining
                                          masks are used.
            train_dataloader_kwargs <dict>: Keyword arguments for the train DataLoader.
                                          Default {'batch_size': 1, 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
            testval_dataloader_kwargs <dict>: Keyword arguments for the test and validation DataLoaders.
                                          Default {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True, 'drop_last': True}
        """
        self.model_mgr_kwargs_list = kwargs.get('model_mgr_kwargs_list')
        self.iterations = kwargs.get('iterations', 5)
        self.model_mgr_kwargs_tweaks = kwargs.get('model_mgr_kwargs_tweaks', [])
        self.metric = kwargs.get('metric', metrics.dice_coeff_metric)
        self.earlystopping_kwargs = kwargs.get('earlystopping_kwargs', dict(min_delta=1e-3, patience=2))
        self.warm_start = kwargs.get('warm_start', None)
        self.overall_best_models = kwargs.get('overall_best_models', False)
        self.dir_checkpoints = kwargs.get('dir_checkpoints', 'checkpoints')
        self.thresholds = kwargs.get('thresholds', dict(disagreement=(.2, .8)))
        self.cot_mask_extension = kwargs.get('cot_mask_extension', '.cot.mask.png')
        self.plots_saving_path = kwargs.get('plots_saving_path', 'plots')
        self.strategy_postprocessing = kwargs.get('strategy_postprocessing', {})
        self.general_postprocessing = kwargs.get('general_postprocessing', [])
        self.postprocessing_threshold = kwargs.get('postprocessing_threshold', .5)

        assert isinstance(self.model_mgr_kwargs_list, (list, tuple)), type(self.model_mgr_kwargs_list)
        assert len(self.model_mgr_kwargs_list) == 2,  'len(self.model_mgr_kwargs_list) != 2'

        for mgr in self.model_mgr_kwargs_list:
            assert isinstance(mgr, dict), type(mgr)

        assert isinstance(self.iterations, int), type(self.iterations)
        assert self.iterations > 0, self.iterations
        assert isinstance(self.model_mgr_kwargs_tweaks, (list, tuple)), type(self.model_mgr_kwargs_tweaks)
        if self.model_mgr_kwargs_tweaks:
            assert len(self.model_mgr_kwargs_tweaks) == len(self.model_mgr_kwargs_list)
        assert callable(self.metric), 'metric must be a callable'
        assert isinstance(self.earlystopping_kwargs, dict), type(self.earlystopping_kwargs)
        if self.warm_start is not None:
            assert isinstance(self.warm_start, dict), type(self.warm_start)
        assert isinstance(self.overall_best_models, bool), type(self.overall_best_models)
        assert isinstance(self.dir_checkpoints, str), type(self.dir_checkpoints)
        if not os.path.isdir(self.dir_checkpoints):
            os.makedirs(self.dir_checkpoints)
        assert isinstance(self.thresholds, (dict)), type(self.thresholds)
        for v in self.thresholds.values():
            v = v if isinstance(v, (list, tuple)) else [v]
            for value in v:
                assert 0 < value < 1, f'{v} is not in range ]0, 1['
        if 'disagreement' in self.thresholds:
            assert self.thresholds['disagreement'][0] < self.thresholds['disagreement'][1], \
                f'{self.thresholds["disagreement"][0]} must be lower than {self.thresholds["disagreement"][1]}'
        assert isinstance(self.cot_mask_extension, str), type(self.cot_mask_extension)
        assert self.cot_mask_extension != '', 'cot_mask_extension cannot be an empty string'
        assert isinstance(self.plots_saving_path, str), type(self.plots_saving_path)
        if self.strategy_postprocessing:
            assert isinstance(self.strategy_postprocessing, dict), type(self.strategy_postprocessing)
            for algorithm_list in self.strategy_postprocessing.values():
                assert isinstance(algorithm_list, (list, tuple)), type(algorithm_list)
                for algorithm in algorithm_list:
                    assert callable(algorithm), f'{algorithm} must be a callable'

        assert isinstance(self.general_postprocessing, list), type(self.general_postprocessing)

        for postprocessing in self.general_postprocessing:
            assert callable(postprocessing), f'{postprocessing} must be a callable'

        assert isinstance(self.postprocessing_threshold, float), type(self.postprocessing_threshold)

        self._SubDatasetsMixin__init(**kwargs)
        self._TorchMetricsBaseMixin__init(**kwargs)

        # Initializing training dataset metrics ###############################
        self.train_combined_preds_metrics = self.train_metrics.clone(prefix='train_combined_preds_metrics_')
        self.train_models_metrics = [self.train_metrics.clone(prefix=f"train_{data['model'].__name__}_")
                                     for data in self.model_mgr_kwargs_list]
        self.train_new_masks_metrics = self.train_metrics.clone(prefix='train_new_masks_metrics_')

        # Initializing validation dataset metrics #############################
        self.valid_combined_preds_metrics = self.valid_metrics.clone(prefix='valid_combined_preds_metrics_')
        self.valid_models_metrics = [self.valid_metrics.clone(prefix=f"valid_{data['model'].__name__}_")
                                     for data in self.model_mgr_kwargs_list]
        self.model_mgr_list = []

    def __call__(self):
        self.process()
        self.test()

    @staticmethod
    def shrink_perturb(modelmgr, lamda=0.5, sigma=0.01):
        """
        Modifies the model weights (shink and/or perturb)

        Kwargs:
            modelmgr <ModelMGR>: ModelMGR instans with pre-loaded weights
            lamda  <float>: Shrink factor in range [0, 1]. Default 0.5
            sigma  <float>: Standard deviation for the Gaussian noise with mean zero to be added. Default 0.01

        Source https://pureai.com/articles/2021/02/01/warm-start-ml.aspx
        """
        assert isinstance(modelmgr, ModelMGR), type(modelmgr)
        assert isinstance(lamda, float), type(lamda)
        assert isinstance(sigma, float), type(sigma)

        if lamda == sigma == .0:
            return

        for (name, param) in modelmgr.module.named_parameters():
            if 'weight' in name:
                if lamda:
                    param.data *= lamda
                if sigma:
                    param.data += torch.normal(0.0, sigma, size=param.shape).to(modelmgr.device)

    def update_model_mgr_kwargs_list(self, iteration):
        """
        Updates the values used to create the model managers after the first iteration (zero-based)
        It should be used to modify/stabilize the training process when using warm_start = True.
        Otherwise, the models could learn too fast or too slow to make the most of the loaded
        pre-trained models. Another option is setting warm_start = False

        Kwargs:
            iteration <int>: zero-based current iteration
        """
        if iteration == 1:
            for data, tweaks in zip(self.model_mgr_kwargs_list, self.model_mgr_kwargs_tweaks):
                data.update(tweaks)

    def create_model_mgr_list(self, data_logger, iteration):
        """
        Deletes old ModelMGR instances from self.model_mgr_list and realeases the GPU cache,
        then creates new instances of ModelMGR and place them into self.model_mgr_list.
        If a warm_start configuration has been provided, it is applied.

        Kwargs:
            data_logger <dict>: dict containing the tracked data
            iteration <int>: zero-based current iteration
        """
        assert isinstance(data_logger, dict), type(data_logger)

        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        self.model_mgr_list.clear()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        self.update_model_mgr_kwargs_list(iteration)

        for kwargs, best_model in zip(self.model_mgr_kwargs_list, self.get_current_best_models(data_logger)):
            model_mgr = ModelMGR(**copy.deepcopy(kwargs))

            if self.warm_start is not None:
                # workaround to avoid errors in the very first iteration where no best models
                # weights exists yet
                try:
                    model_mgr.load(best_model)
                except AssertionError:
                    pass
                else:
                    self.shrink_perturb(model_mgr, **self.warm_start)

            self.model_mgr_list.append(model_mgr)

    def remove_cot_masks(self):
        """ Removes all cotraining masks """
        subprocess.run(
            f'find {self.dataset_kwargs["train_path"]} -name *{self.cot_mask_extension} -delete'.split(),
            cwd='.', check=True
        )

    def set_models_to_eval_mode(self):
        """ Sets the models to eval mode """
        for model_mgr in self.model_mgr_list:
            model_mgr.module.eval()

    def set_models_to_train_mode(self):
        """ Sets the models to train mode """
        for model_mgr in self.model_mgr_list:
            model_mgr.module.train()

    def load_best_models(self, data_logger):
        """
        Load the best models only for inference

        Kwargs:
            data_logger <dict>: dict containing the tracked data
        """
        assert isinstance(data_logger, dict), type(data_logger)

        for model_mgr, best_model in zip(self.model_mgr_list, self.get_current_best_models(data_logger)):
            model_mgr.load(best_model)

    def train_models(self):
        """ Trains the models """
        # TODO: maybe I should train the models in different adversarial datasets generated
        #       from the same train dataset (this would be our views), and apply the disagreement
        #       strategy when comparing the predictions done on the original train dataset.
        #       This could be considered as a way consistency regularization
        for idx, mgr in enumerate(self.model_mgr_list, start=1):
            logger.info(f'TRAINING MODEL {idx}: {mgr.module.__class__.__name__}')
            mgr()

    @staticmethod
    def apply_mask_postprocessing(postprocessing_list, sub_preds, /, **kwargs):
        """
        Applies the postprocessing algorithms to sub_preds in an incremental way.

        NOTE: Use the extra keyword arguments to pass data to the postprocessing methods

        Kwargs:
            postprocessing_list <list, tuple>: List of callable postprocessing algorithms
            sub_preds <torch.Tensor>: Masks [..., H, W] containing the disagreement or agreement
                                      pixels or any other of filtered pixels that may or may not
                                      have some intersections with the original masks

        Returns:
            postprocessed_sub_preds <torch.Tensor>
        """
        assert isinstance(postprocessing_list, (list, tuple)), type(postprocessing_list)
        assert isinstance(sub_preds, torch.Tensor), type(sub_preds)

        for postprocessing in postprocessing_list:
            assert callable(postprocessing), f'{postprocessing} must be a callable'

        for postprocessing in postprocessing_list:
            sub_preds = postprocessing(sub_preds, **kwargs)

        return sub_preds

    def agreement(self, results):
        """
        Calculates new mask values from each model using agreement strategy

        Kwargs:
            results <list>: List of mask predicted by the models

        Returns:
            new_mask_values <torch.Tensor>
        """
        assert isinstance(results, list), type(results)

        new_mask_values_from_model_1 = (results[0] > self.thresholds['agreement']).float()
        new_mask_values_from_model_2 = (results[1] > self.thresholds['agreement']).float()

        return new_mask_values_from_model_1 * new_mask_values_from_model_2

    def disagreement(self, results):
        """
        Calculates new mask values from each model using disagreement strategy

        Kwargs:
            results <list>: List of mask predicted by the models

        Returns:
            new_mask_values <torch.Tensor>
        """
        assert isinstance(results, list), type(results)

        new_mask_values_from_model_1 = (
            (results[0] > self.thresholds['disagreement'][1]) *
            (results[1] < self.thresholds['disagreement'][0])
        ).float()
        new_mask_values_from_model_2 = (
            (results[1] > self.thresholds['disagreement'][1]) *
            (results[0] < self.thresholds['disagreement'][0])
        ).float()

        return new_mask_values_from_model_1.max(new_mask_values_from_model_2)

    def selfcombined(self, results):
        """
        Calculates new mask values from each model using self-combined strategy

        Kwargs:
            results <list>: List of mask predicted by the models

        Returns:
            new_mask_values <torch.Tensor>
        """
        assert isinstance(results, list), type(results)

        new_mask_values_from_model_1 = (results[0] > self.thresholds['selfcombined']).float()
        new_mask_values_from_model_2 = (results[1] > self.thresholds['selfcombined']).float()

        return new_mask_values_from_model_1.max(new_mask_values_from_model_2)

    def get_new_mask_values(self, results, /, **kwargs):
        """
        Calculates and returns the new masks values using the stategies especified in self.threholds

        NOTE: Use the extra keyword arguments to pass data to the postprocessing methods

        Kwargs:
            results <list>: List of masks predicted by the models

        Returns:
            new_mask_values <torch.Tensor>
        """
        new_mask_values = torch.zeros_like(results[0]).to(results[0].device)

        for key in self.thresholds:
            strategy_new_mask_values = getattr(self, key)(results)
            # applying all the mask postprocessing algorithms for the current strategy in
            # an incremental way
            strategy_new_mask_values = self.apply_mask_postprocessing(
                self.strategy_postprocessing.get(key, []),
                strategy_new_mask_values,
                **kwargs
            )
            new_mask_values = new_mask_values.max(strategy_new_mask_values)

        return new_mask_values

    def get_combined_predictions(self, results):
        """
        Calculates new mask values from each model using self-combined strategy with
        a used-defined threshold. These masks are intended for mask postprocessing purposes

        Kwargs:
            results <list>: List of mask predicted by the models

        Returns:
            combined_preds <torch.Tensor>
        """
        assert isinstance(results, list), type(results)

        new_mask_values_from_model_1 = (results[0] > self.postprocessing_threshold).float()
        new_mask_values_from_model_2 = (results[1] > self.postprocessing_threshold).float()

        return new_mask_values_from_model_1.max(new_mask_values_from_model_2)

    def strategy(self) -> Tuple[dict, dict, List[dict], List[torch.Tensor]]:
        """
        Performs the strategy to update/create the co-training ground truth masks

        Returns:
            new_masks_metrics<dict>, combined_preds_metrics<dict>, models_metrics<List[dict]>,
            models_losses<List[torch.Tensor]>
        """
        self.set_models_to_eval_mode()

        results = None
        models_metrics = [0] * len(self.model_mgr_kwargs_list)
        models_losses = copy.deepcopy(models_metrics)
        total_batches = len(self.train_loader)

        for batch in tqdm(
                self.train_loader, total=total_batches,
                desc=f'{", ".join(list(self.thresholds.keys())).capitalize()} round',
                unit='batch', disable=DISABLE_PROGRESS_BAR):
            results = []
            model_mask_thresholds = []
            true_masks = None

            for idx, model_mgr in enumerate(self.model_mgr_list):
                # we do not apply the model_mgr threshold because we are going to
                # evaluate the difference of the scores against the self.thresholds[<strategy>]
                loss, extra_data = model_mgr.validation_step(batch=batch, apply_threshold=False)
                results.append(extra_data['pred'])
                true_masks = extra_data['true_masks']

                # properly calculating the model metric using its corresponding mask_threshold
                self.train_models_metrics[idx].update(
                    (extra_data['pred'] > model_mgr.mask_threshold).float(),
                    true_masks
                )

                model_mask_thresholds.append(model_mgr.mask_threshold)
                models_losses[idx] += loss

            self.train_combined_preds_metrics.update(
                (results[0].max(results[1]) > min(model_mask_thresholds)).float(),
                true_masks
            )

            # creating new combined pred masks (for mask post-processing)
            combined_pred_masks = self.get_combined_predictions(results)
            new_masks = self.get_new_mask_values(results, preds=combined_pred_masks)

            # applying all the mask postprocessing algorithms in an incremental way
            new_masks = self.apply_mask_postprocessing(
                self.general_postprocessing, new_masks, preds=combined_pred_masks)

            # creating final new masks by contatenating true masks (co-training masks) or
            # the original ground truth masks (if available) with the new mask values
            # selected by the especified strategies and postprocessing approaches
            if isinstance(batch['original_mask'], torch.Tensor):
                new_masks = batch['original_mask'].to(new_masks.device).max(new_masks)
            else:
                new_masks = true_masks.max(new_masks)

            self.train_new_masks_metrics.update(new_masks, true_masks)

            # saving new masks
            for new_mask, mask_path in zip(new_masks, batch['updated_mask_path']):
                new_mask = Image.fromarray(new_mask.squeeze().detach().cpu().numpy() * 255).convert('L')
                new_mask.save(mask_path)

        for idx in range(len(self.model_mgr_list)):
            models_metrics[idx] = self.train_models_metrics[idx].compute()
            self.train_models_metrics[idx].reset()
            models_losses[idx] /= total_batches

        self.set_models_to_train_mode()
        # total metrics over all training batches
        combined_preds_metrics = self.train_combined_preds_metrics.compute()
        new_masks_metrics = self.train_new_masks_metrics.compute()
        # reset metrics states after each epoch
        self.train_combined_preds_metrics.reset()
        self.train_new_masks_metrics.reset()

        return new_masks_metrics, combined_preds_metrics, models_metrics, models_losses

    def validation(self, testing: bool = False) -> Tuple[float, dict, list, list]:
        """
        Kwargs:
            testing <bool>: If True uses the test_loader; otherwise, it uses val_loader

        Returns:
            new_masks_metrics<float>, combined_preds_metrics<dict>, models_metrics<list>, models_losses<list>
        """
        assert isinstance(testing, bool), type(testing)

        self.set_models_to_eval_mode()

        results = None
        models_metrics = [0] * (len(self.model_mgr_list))
        models_losses = copy.deepcopy(models_metrics)
        new_masks_metrics = 0  # we're not calculating new_masks_metrics for validation/testing

        if testing:
            dataloader, desc, total_batches = self.test_loader, 'Testing', len(self.test_loader)
        else:
            dataloader, desc, total_batches = self.val_loader, 'Validation', len(self.val_loader)

        for batch in tqdm(dataloader, total=total_batches, desc=f'{desc} round', unit='batch',
                          disable=DISABLE_PROGRESS_BAR):
            results = []
            true_masks = None

            for idx, model_mgr in enumerate(self.model_mgr_list):
                # we are not going to use the self.thresholds for any operation
                # so we can apply the model_mgr threshold
                loss, extra_data = model_mgr.validation_step(batch=batch, apply_threshold=True)
                results.append(extra_data['pred'])
                true_masks = extra_data['true_masks']
                self.valid_models_metrics[idx].update(
                    (extra_data['pred'] > model_mgr.mask_threshold).float(),
                    true_masks
                )
                models_losses[idx] += loss

            self.valid_combined_preds_metrics.update(results[0].max(results[1]), true_masks)

        for idx in range(len(self.model_mgr_list)):
            models_metrics[idx] = self.valid_models_metrics[idx].compute()
            self.valid_models_metrics[idx].reset()
            models_losses[idx] /= total_batches

        self.set_models_to_train_mode()

        # total metrics over all training batches
        combined_preds_metrics = self.valid_combined_preds_metrics.compute()
        # reset metrics states after each epoch
        self.valid_combined_preds_metrics.reset()

        return new_masks_metrics, combined_preds_metrics, models_metrics, models_losses

    @timing
    def test(self):
        """ Performs the testing using the provided subdataset """
        if self.test_loader is None:
            logger.error("No testing dataloader was provided")
            return

        new_masks_metric, combined_preds_metric, models_metrics, _ = self.validation(testing=True)
        logger.info(
            f'Testing New Masks Metric: {new_masks_metric:.6f} \t'
            f'Combined Preds Metric: {combined_preds_metric:.6f} \t'
            f', '.join([f'model {idx}: {val:.6f}' for idx, val in enumerate(
                models_metrics, start=1)])
        )

    def get_current_best_models(self, data_logger: dict) -> Union[List[OrderedDict], List[str]]:
        """
        Finds and returns the overall best state dicts of the models. If it is the very first
        iteration or self.overall_best_models is False, it returns a list of emtpy strings

        Kwargs:
            data_logger <dict>: dict containing the tracked data

        Returns:
            [state_dict<OrderedDict>, state_dict<OrderedDict>, ...] or ['', '', ...]
        """
        assert isinstance(data_logger, dict), type(data_logger)

        val_models_metrics = np.array([list(map(self.get_mean_main_metrics, metrics))
                                       for metrics in data_logger['val_models_metrics']])
        if not self.overall_best_models or not val_models_metrics.size:
            return [''] * len(self.model_mgr_kwargs_list)

        best_models = []

        for model_idx, _ in enumerate(self.model_mgr_kwargs_list):
            best_cot_iter = val_models_metrics[:, model_idx].argmax()

            # Verifying if the current models are the best ones
            # but first asking if the all the ModelMGR instances have been already created
            if len(self.model_mgr_list) == len(self.model_mgr_kwargs_list):
                best_chkpt_data = self.model_mgr_list[model_idx].get_best_checkpoint_data()
                model_best_metric = self.get_best_combined_main_metrics(
                    best_chkpt_data['data_logger']['val_metric'])

                if model_best_metric > val_models_metrics[:, model_idx][best_cot_iter]:
                    best_models.append(best_chkpt_data.pop('model_state_dict'))
                    continue

            # getting best models from cotraining iterations checkpoints
            data = torch.load(os.path.join(self.dir_checkpoints, self.checkpoint_pattern.format(best_cot_iter)))
            best_models.append(data.pop(f'model{model_idx+1}'))

        return best_models

    def save(self, iteration: int, data_logger: dict):
        """
        Saves data logger and models from the current iteration

        Kwargs:
            iteration    <int>: iteration number
            data_logger <dict>: dict containing the tracked data
        """
        assert isinstance(iteration, int), type(iteration)
        assert isinstance(data_logger, dict), type(data_logger)

        data = dict(
            iteration=iteration,
            data_logger=data_logger,
            model1=self.model_mgr_list[0].model.state_dict(),
            model2=self.model_mgr_list[1].model.state_dict()
        )

        torch.save(
            data,
            os.path.join(self.dir_checkpoints, self.checkpoint_pattern.format(data['iteration']))
        )

    @timing
    def process(self):
        """ Performs the whole co-training process """
        data_logger = dict(
            train_new_masks_metric=[], train_combined_preds_metric=[], train_models_metrics=[],
            train_models_losses=[],
            val_new_masks_metric=[], val_combined_preds_metric=[], val_models_metrics=[],
            val_models_losses=[],
        )
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        previous_train_new_masks_metric = new_train_new_masks_metric = .0

        self.remove_cot_masks()

        for i in range(self.iterations):
            logger.info(f'CO-TRAINING: ITERATION {i+1}')

            self.create_model_mgr_list(data_logger, i)
            self.train_models()
            self.load_best_models(data_logger)

            new_masks_metric, combined_preds_metric, models_metrics, models_losses = self.strategy()
            data_logger['train_new_masks_metric'].append(self.prepare_to_save(new_masks_metric))
            data_logger['train_combined_preds_metric'].append(self.prepare_to_save(combined_preds_metric))
            data_logger['train_models_metrics'].append(self.prepare_to_save(models_metrics))
            data_logger['train_models_losses'].append(self.prepare_to_save(models_losses))

            new_masks_metric, combined_preds_metric, models_metrics, models_losses = self.validation()
            data_logger['val_new_masks_metric'].append(new_masks_metric)
            data_logger['val_combined_preds_metric'].append(self.prepare_to_save(combined_preds_metric))
            data_logger['val_models_metrics'].append(self.prepare_to_save(models_metrics))
            data_logger['val_models_losses'].append(self.prepare_to_save(models_losses))

            self.print_epoch_summary(data_logger, i)
            self.save(i, data_logger)

            new_train_new_masks_metric = self.get_mean_main_metrics(data_logger['train_new_masks_metric'][i])

            if earlystopping(min(previous_train_new_masks_metric, new_train_new_masks_metric),
                             max(previous_train_new_masks_metric, new_train_new_masks_metric)):
                break

            previous_train_new_masks_metric = new_train_new_masks_metric
