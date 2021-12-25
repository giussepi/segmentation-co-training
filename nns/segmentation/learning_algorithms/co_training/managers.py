# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/managers """

import os
import copy
import subprocess

import torch
from gutils.decorators import timing
from gtorch_utils.segmentation import metrics
from logzero import logger
from PIL import Image
from tqdm import tqdm

from nns.callbacks.plotters.training import TrainingPlotter
from nns.managers import ModelMGR
from nns.mixins.subdatasets import SubDatasetsMixin


class CoTraining(SubDatasetsMixin):
    """

    Usage:
        CoTraining([modelmgr1, modelgr2, ...])

    """

    checkpoint_pattern = 'chkpt_{}.pth.tar'

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            model_mgr_kwargs_list <list, tuple>: list or tuple containing dictionaries to create instances of
                                          ModelMGR. Two dictionaries are required.
            iterations             <int>: Number of iterations to execute the co-training. Default 5
            metric            <callable>: Metric to measure the results. See gtorch_utils.segmentation.metrics
                                          Default metrics.dice_coeff_metric
            warm_start            <bool>: If True new model instances will be initialized with the best
                                          models weights found previously. Default False
            dir_checkpoints        <str>: path to the directory where checkpoints will be saved
            thresholds     <list, tuple>: list containing [low, high] thresholds to select mask values
                                          and perform the disagreement strategy.
                                          E.g. Selecting the new mask values from model1 can be done by
                                          (model1 > thresholds[1]) * (model2 < threshold[0]).
                                          Default (.2, .8)
            cot_mask_extension     <str>: co-traning mask extension. Default '.cot.mask.png'
            plots_saving_path      <str>: Path to the folder used to save the plots. Default 'plots'
            ###################################################################
            #                         SubDatasetsMixin                        #
            ###################################################################
            dataset (DatasetTemplate): Custom dataset class descendant of gtorch_utils.datasets.segmentation.DatasetTemplate.
                See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
            dataset_kwargs (dict): keyword arguments for the dataset. Default {}
            train_dataloader_kwargs <dict>: Keyword arguments for the train DataLoader.
                Default {'batch_size': 1, 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
            testval_dataloader_kwargs <dict>: Keyword arguments for the test and validation DataLoaders.
                Default {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True, 'drop_last': True}
        """
        self.model_mgr_kwargs_list = kwargs.get('model_mgr_kwargs_list')
        self.iterations = kwargs.get('iterations', .8)
        self.metric = kwargs.get('metric', metrics.dice_coeff_metric)
        self.warm_start = kwargs.get('warm_start', False)
        self.dir_checkpoints = kwargs.get('dir_checkpoints', 'checkpoints')
        self.thresholds = kwargs.get('thresholds', (.2, .8))
        self.cot_mask_extension = kwargs.get('cot_mask_extension', '.cot.mask.png')
        self.plots_saving_path = kwargs.get('plots_saving_path', 'plots')
        # TODO: todo add a min/max value to be achieved by the metric as an alternative stopping criteria
        assert isinstance(self.model_mgr_kwargs_list, (list, tuple)), type(self.model_mgr_kwargs_list)
        assert len(self.model_mgr_kwargs_list) == 2,  'len(self.model_mgr_kwargs_list) != 2'

        for mgr in self.model_mgr_kwargs_list:
            # assert isinstance(mgr, ModelMGR), type(mgr)
            assert isinstance(mgr, dict), type(mgr)

        assert isinstance(self.iterations, int), type(self.iterations)
        assert self.iterations > 0, self.iterations
        assert callable(self.metric), 'metric must be a callable'
        assert isinstance(self.warm_start, bool), type(self.warm_start)
        assert isinstance(self.dir_checkpoints, str), type(self.dir_checkpoints)
        if not os.path.isdir(self.dir_checkpoints):
            os.makedirs(self.dir_checkpoints)
        assert isinstance(self.thresholds, (list, tuple)), f'thresholds must be a list or tuple'
        assert len(self.thresholds) == 2, f'{len(self.thresholds) != 2}'
        assert 0 < self.thresholds[0] < 1, f'{self.thresholds[0]} is not in range ]0, 1['
        assert 0 < self.thresholds[1] < 1, f'{self.thresholds[1]} is not in range ]0, 1['
        assert self.thresholds[0] < self.thresholds[1], f'{self.thresholds[0]} must be lower than {self.thresholds[1]}'

        assert isinstance(self.cot_mask_extension, str), type(self.cot_mask_extension)
        assert self.cot_mask_extension != '', 'cot_mask_extension cannot be an empty string'

        self.init_SubDatasetsMixin(**kwargs)
        self.model_mgr_list = []

    def __call__(self):
        self.process()
        self.test()

    def create_model_mgr_list(self):
        """
        Deletes old ModelMGR instances from self.model_mgr_list and realeases the GPU cache,
        then creates new instances of ModelMGR and place them into self.model_mgr_list
        """
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        self.model_mgr_list.clear()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

        for kwargs in self.model_mgr_kwargs_list:
            model_mgr = ModelMGR(**copy.deepcopy(kwargs))

            if self.warm_start:
                # workaround to avoid errors in the very first iteration where no best models
                # weights exists yet
                try:
                    model_mgr.load()
                except AssertionError:
                    pass

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
            model_mgr.model.eval()

    def set_models_to_train_mode(self):
        """ Sets the models to train mode """
        for model_mgr in self.model_mgr_list:
            model_mgr.model.train()

    def load_best_models(self):
        """ Load the best models only for inference """
        for model_mgr in self.model_mgr_list:
            model_mgr.load()

    def train_models(self):
        """ Trains the models """
        # TODO: maybe I should train the models in different adversarial datasets generated
        #       from the same train dataset (this would be our views), and apply the disagreement
        #       strategy when comparing the predictions done on the original train dataset.
        #       This could be considered as a way consistency regularization
        for idx, mgr in enumerate(self.model_mgr_list, start=1):
            logger.info(f'TRAINING MODEL {idx}')
            mgr()

    def disagreement(self):
        """
        Performs the disagreement strategy to update/create the co-training ground truth masks

        Returns:
            new_masks_metric<float>, combined_preds_metric<float>,  models_metrics<list>, models_losses<list>
        """
        self.set_models_to_eval_mode()

        results = None
        models_metrics = [0] * (len(self.model_mgr_kwargs_list))
        models_losses = copy.deepcopy(models_metrics)
        combined_preds_metric = new_masks_metric = 0
        total_batches = len(self.train_loader)

        for batch in tqdm(self.train_loader, total=total_batches, desc='Disagreement round', unit='batch'):
            results = []
            model_mask_thresholds = []
            true_masks = None

            for idx, model_mgr in enumerate(self.model_mgr_list):
                # we do not apply the model_mgr threshold because we are going to
                # evaluate the difference of the scores against the self.thresholds
                loss, _, _, extra_data = model_mgr.validation_step(batch=batch, apply_threshold=False)
                results.append(extra_data['pred'])
                true_masks = extra_data['true_masks']

                # properly calculating the model metric using its corresponding mask_threshold
                models_metrics[idx] += self.metric(
                    (extra_data['pred'] > model_mgr.mask_threshold).float(),
                    true_masks
                ).item()
                model_mask_thresholds.append(model_mgr.mask_threshold)
                models_losses[idx] += loss

            combined_preds_metric += self.metric(
                (results[0].max(results[1]) > min(model_mask_thresholds)).float(),
                true_masks
            ).item()
            new_mask_values_from_model_1 = (
                (results[0] > self.thresholds[1]) * (results[1] < self.thresholds[0])
            ).float()
            new_mask_values_from_model_2 = (
                (results[1] > self.thresholds[1]) * (results[0] < self.thresholds[0])
            ).float()

            # creating new masks using selected values from model 1 and model 2
            new_masks = true_masks.max(new_mask_values_from_model_1).max(new_mask_values_from_model_2)
            # TODO: Stopping Criteria
            # new_masks_metric can be used as a stopping condition with a user-defined threshold
            # so if the metric between the old masks (true_masks) and the new ones does not
            # change more than the user-defined threshold we can stop whole process
            new_masks_metric += self.metric(new_masks, true_masks).item()  # does not make sense, it is 1

            for new_mask, mask_path in zip(new_masks, batch['updated_mask_path']):
                new_mask = Image.fromarray(new_mask.squeeze().detach().cpu().numpy() * 255).convert('L')
                new_mask.save(mask_path)

        for idx in range(len(self.model_mgr_list)):
            models_metrics[idx] /= total_batches
            models_losses[idx] /= total_batches

        self.set_models_to_train_mode()

        return new_masks_metric / total_batches, combined_preds_metric / total_batches, models_metrics, \
            models_losses

    def validation(self, testing=False):
        """
        Kwargs:
            testing <bool>: If True uses the test_loader; otherwise, it uses val_loader

        Returns:
            new_masks_metric<float>, combined_preds_metric<float>,  models_metrics<list>, models_losses<list>
        """
        assert isinstance(testing, bool), type(testing)

        self.set_models_to_eval_mode()

        results = None
        models_metrics = [0] * (len(self.model_mgr_list))
        models_losses = copy.deepcopy(models_metrics)
        combined_preds_metric = new_masks_metric = 0

        if testing:
            dataloader, desc, total_batches = self.test_loader, 'Testing', len(self.test_loader)
        else:
            dataloader, desc, total_batches = self.val_loader, 'Validation', len(self.val_loader)

        for batch in tqdm(dataloader, total=total_batches, desc=f'{desc} round', unit='batch'):
            results = []
            true_masks = None

            for idx, model_mgr in enumerate(self.model_mgr_list):
                # we are not going to use the self.thresholds for any operation
                # so we wan apply the model_mgr threshol  and directly use its returned metric
                loss, metric, _, extra_data = model_mgr.validation_step(batch=batch, apply_threshold=True)
                results.append(extra_data['pred'])
                true_masks = extra_data['true_masks']
                models_metrics[idx] += metric
                models_losses[idx] += loss

            combined_preds_metric += self.metric(results[0].max(results[1]), true_masks).item()

        for idx in range(len(self.model_mgr_list)):
            models_metrics[idx] /= total_batches
            models_losses[idx] /= total_batches

        self.set_models_to_train_mode()

        return new_masks_metric / total_batches, combined_preds_metric / total_batches, models_metrics, \
            models_losses

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

    def save(self, iteration, data_logger):
        """
        Saves data logger and best models in the current iteration

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

        self.remove_cot_masks()

        for i in range(self.iterations):
            logger.info(f'CO-TRAINING: ITERATION {i+1}')

            self.create_model_mgr_list()
            self.train_models()
            self.load_best_models()

            new_masks_metric, combined_preds_metric, models_metrics, models_losses = self.disagreement()
            data_logger['train_new_masks_metric'].append(new_masks_metric)
            data_logger['train_combined_preds_metric'].append(combined_preds_metric)
            data_logger['train_models_metrics'].append(models_metrics)
            data_logger['train_models_losses'].append(models_losses)

            new_masks_metric, combined_preds_metric, models_metrics, models_losses = self.validation()
            data_logger['val_new_masks_metric'].append(new_masks_metric)
            data_logger['val_combined_preds_metric'].append(combined_preds_metric)
            data_logger['val_models_metrics'].append(models_metrics)
            data_logger['val_models_losses'].append(models_losses)

            logger.info(
                f'Co-training iteration {i+1}:\n'
                f'train_new_masks_metric: {data_logger["train_new_masks_metric"][i]:.6f} \t'
                f'train_combined_preds_metric: {data_logger["train_combined_preds_metric"][i]:.6f} \t'
                f'train_models_metrics: ' + f', '.join([f'model {idx}: {val:.6f}' for idx, val in enumerate(
                    data_logger["train_models_metrics"][i], start=1)]) + '\t'
                f'train_models_losses: ' + f', '.join([f'model {idx}: {val:.6f}' for idx, val in enumerate(
                    data_logger["train_models_losses"][i], start=1)]) +
                f'\n val_new_masks_metric: {data_logger["val_new_masks_metric"][i]:.6f} \t'
                f'val_combined_preds_metric: {data_logger["val_combined_preds_metric"][i]:.6f} \t'
                f'val_models_metrics: ' + f', '.join([f'model {idx}: {val:.6f}' for idx, val in enumerate(
                    data_logger["val_models_metrics"][i], start=1)]) + '\t'
                f'val_models_losses: ' + f', '.join([f'model {idx}: {val:.6f}' for idx, val in enumerate(
                    data_logger["val_models_losses"][i], start=1)])
            )

            self.save(i, data_logger)

    def plot_and_save(self, checkpoint, save=False, dpi='figure', show=True):
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
        for idx, _ in enumerate(self.model_mgr_kwargs_list):
            TrainingPlotter(
                train_loss=torch.as_tensor(data['train_models_losses'])[:, idx].cpu().tolist(),
                train_metric=torch.as_tensor(data['train_models_metrics'])[:, idx].cpu().tolist(),
                val_loss=torch.as_tensor(data['val_models_losses'])[:, idx].cpu().tolist(),
                val_metric=torch.as_tensor(data['val_models_metrics'])[:, idx].cpu().tolist()
            )(
                lm_title=f'Model {idx+1}: Metrics and Losses',
                xlabel='Co-training iterations',
                lm_ylabel='Loss and Metric',
                lm_legend_kwargs=dict(shadow=True, fontsize=8, loc='best'),
                lm_saving_path=os.path.join(self.plots_saving_path, f'model_{idx+1}: losses_metrics.png'),
                save=save,
                dpi=dpi,
                show=show,
            )

        # plotting new masks metrics and combined preds metrics
        TrainingPlotter(
            train_loss=data['train_new_masks_metric'],
            train_metric=data['train_combined_preds_metric'],
            val_loss=data['val_new_masks_metric'],
            val_metric=data['val_combined_preds_metric']
        )(
            lm_title='New masks and combined masks metrics',
            xlabel='Co-training iterations',
            lm_ylabel='Metric',
            lm_legend_kwargs=dict(shadow=True, fontsize=8, loc='best'),
            lm_saving_path=os.path.join(self.plots_saving_path, 'new_masks_combined_preds.png'),
            train_loss_label='train new masks',
            val_loss_label='val new masks',
            train_metric_label='train combined preds',
            val_metric_label='val combined preds',
            save=save,
            dpi=dpi,
            show=show
        )
