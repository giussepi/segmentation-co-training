# -*- coding: utf-8 -*-
""" nns/segmentation/learning_algorithms/co_training/managers """

import os
import copy
import subprocess

import torch
import numpy as np
from gutils.decorators import timing
from gtorch_utils.nns.managers.callbacks import EarlyStopping
from gtorch_utils.segmentation import metrics
from logzero import logger
from PIL import Image
from tabulate import tabulate
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
            earlystopping_kwargs  <dict>: Early stopping configuration.
                                          Default dict(min_delta=1e-3, patience=2)
            warm_start      <dict, None>: Configuration of the warm start. Set it to a dict full of zeroes to
                                          only load the weights (e.g. {'lamda': .0, 'sigma': .0}).
                                          Set it to None to not perform warm start. Default None.
            dir_checkpoints        <str>: path to the directory where checkpoints will be saved
            thresholds            <dict>: Dictionary containing as keys the strategies to apply, and as values
                                          their thresholds. E.g. dict(agreement=.9, disagreement=(.2, .8)) or
                                          dict(selfcombined=.9). Default dict(disagreement=(.2, .8))
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
        self.iterations = kwargs.get('iterations', 5)
        self.metric = kwargs.get('metric', metrics.dice_coeff_metric)
        self.earlystopping_kwargs = kwargs.get('earlystopping_kwargs', dict(min_delta=1e-3, patience=2))
        self.warm_start = kwargs.get('warm_start', None)
        self.dir_checkpoints = kwargs.get('dir_checkpoints', 'checkpoints')
        self.thresholds = kwargs.get('thresholds', dict(disagreement=(.2, .8)))
        self.cot_mask_extension = kwargs.get('cot_mask_extension', '.cot.mask.png')
        self.plots_saving_path = kwargs.get('plots_saving_path', 'plots')

        assert isinstance(self.model_mgr_kwargs_list, (list, tuple)), type(self.model_mgr_kwargs_list)
        assert len(self.model_mgr_kwargs_list) == 2,  'len(self.model_mgr_kwargs_list) != 2'

        for mgr in self.model_mgr_kwargs_list:
            assert isinstance(mgr, dict), type(mgr)

        assert isinstance(self.iterations, int), type(self.iterations)
        assert self.iterations > 0, self.iterations
        assert callable(self.metric), 'metric must be a callable'
        assert isinstance(self.earlystopping_kwargs, dict), type(self.earlystopping_kwargs)
        if self.warm_start is not None:
            assert isinstance(self.warm_start, dict), type(self.warm_start)
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

        self.init_SubDatasetsMixin(**kwargs)
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

        for (name, param) in modelmgr.model.named_parameters():
            if 'weight' in name:
                if lamda:
                    param.data *= lamda
                if sigma:
                    param.data += torch.normal(0.0, sigma, size=param.shape).to(modelmgr.device)

    def create_model_mgr_list(self):
        """
        Deletes old ModelMGR instances from self.model_mgr_list and realeases the GPU cache,
        then creates new instances of ModelMGR and place them into self.model_mgr_list.
        If a warm_start configuration has been provided, it is applied.
        """
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        self.model_mgr_list.clear()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

        for kwargs in self.model_mgr_kwargs_list:
            model_mgr = ModelMGR(**copy.deepcopy(kwargs))

            if self.warm_start is not None:
                # workaround to avoid errors in the very first iteration where no best models
                # weights exists yet
                try:
                    model_mgr.load()
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
            logger.info(f'TRAINING MODEL {idx}: f{mgr.model.module.__class__.__name__}')
            mgr()

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

    def get_new_mask_values(self, results):
        """
        Calculates and returns the new masks values using the stategies especified in self.threholds

        Kwargs:
            results <list>: List of masks predicted by the models

        Returns:
            new_mask_values <torch.Tensor>
        """
        new_mask_values = torch.zeros_like(results[0]).to(results[0].device)

        for key in self.thresholds:
            strategy_new_mask_values = getattr(self, key)(results)
            new_mask_values = new_mask_values.max(strategy_new_mask_values)

        return new_mask_values

    def strategy(self):
        """
        Performs the strategy to update/create the co-training ground truth masks

        Returns:
            new_masks_metric<float>, combined_preds_metric<float>,  models_metrics<list>, models_losses<list>
        """
        self.set_models_to_eval_mode()

        results = None
        models_metrics = [0] * len(self.model_mgr_kwargs_list)
        models_losses = copy.deepcopy(models_metrics)
        combined_preds_metric = new_masks_metric = 0
        total_batches = len(self.train_loader)

        for batch in tqdm(
                self.train_loader, total=total_batches,
                desc=f'{", ".join(list(self.thresholds.keys())).capitalize()} round',
                unit='batch'):
            results = []
            model_mask_thresholds = []
            true_masks = None

            for idx, model_mgr in enumerate(self.model_mgr_list):
                # we do not apply the model_mgr threshold because we are going to
                # evaluate the difference of the scores against the self.thresholds[<strategy>]
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

            # creating new masks contatenating true masks with the new mask values selected
            # by the especified strategies
            new_masks = true_masks.max(self.get_new_mask_values(results))
            new_masks_metric += self.metric(new_masks, true_masks).item()

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
        earlystopping = EarlyStopping(**self.earlystopping_kwargs)
        previous_train_new_masks_metric = .0

        self.remove_cot_masks()

        for i in range(self.iterations):
            logger.info(f'CO-TRAINING: ITERATION {i+1}')

            self.create_model_mgr_list()
            self.train_models()
            self.load_best_models()

            new_masks_metric, combined_preds_metric, models_metrics, models_losses = self.strategy()
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

            if earlystopping(min(previous_train_new_masks_metric, data_logger['train_new_masks_metric'][i]),
                             max(previous_train_new_masks_metric, data_logger['train_new_masks_metric'][i])):
                break

            previous_train_new_masks_metric = data_logger['train_new_masks_metric'][i]

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
        for idx, mmgr in enumerate(self.model_mgr_kwargs_list):
            TrainingPlotter(
                train_loss=torch.as_tensor(data['train_models_losses'])[:, idx].detach().cpu().tolist(),
                train_metric=torch.as_tensor(data['train_models_metrics'])[:, idx].detach().cpu().tolist(),
                val_loss=torch.as_tensor(data['val_models_losses'])[:, idx].detach().cpu().tolist(),
                val_metric=torch.as_tensor(data['val_models_metrics'])[:, idx].detach().cpu().tolist()
            )(
                lm_title=f'Model {idx+1} ({mmgr["model"].module.__class__.__name__}): Metrics and Losses',
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

    def print_data_logger_summary(self, checkpoint, tablefmt='orgtbl'):
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
        for idx, mmgr in enumerate(self.model_mgr_kwargs_list):
            data = [["key", "Validation", "corresponding training value"]]
            data_logger['val_models_metrics'] = torch.as_tensor(
                data_logger['val_models_metrics']).cpu().numpy()
            data_logger['train_models_metrics'] = torch.as_tensor(
                data_logger['train_models_metrics']).detach().cpu().numpy()
            data_logger['val_models_losses'] = torch.as_tensor(
                data_logger['val_models_losses']).detach().cpu().numpy()
            data_logger['train_models_losses'] = torch.as_tensor(
                data_logger['train_models_losses']).detach().cpu().numpy()

            max_idx = np.argmax(data_logger['val_models_metrics'][:, idx])
            data.append(["Best metric",
                         f"{data_logger['val_models_metrics'][max_idx, idx]:.4f}",
                         f"{data_logger['train_models_metrics'][max_idx, idx]:.4f}"])
            min_idx = np.argmin(data_logger['val_models_losses'][:, idx])
            data.append(["Min loss",
                         f"{data_logger['val_models_losses'][min_idx, idx]:.4f}",
                         f"{data_logger['train_models_losses'][min_idx, idx]:.4f}"])

            print(f'MODEL {idx+1} ({mmgr["model"].module.__class__.__name__}):')
            print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
            print('\n')

        # plotting new masks metrics and combined preds metrics
        data = [["key", "Validation", "corresponding training value"]]
        max_idx = np.argmax(data_logger['val_new_masks_metric'])
        data.append(["New masks best metric",
                     f"{data_logger['val_new_masks_metric'][max_idx]:.4f}",
                     f"{data_logger['train_new_masks_metric'][max_idx]:.4f}"])
        max_idx = np.argmax(data_logger['val_combined_preds_metric'])
        data.append(["Combined preds best metric",
                     f"{data_logger['val_combined_preds_metric'][max_idx]:.4f}",
                     f"{data_logger['train_combined_preds_metric'][max_idx]:.4f}"])

        print('New mask and combined predictions metrics')
        print(tabulate(data, headers="firstrow", showindex=False, tablefmt=tablefmt))
        print('\n')
