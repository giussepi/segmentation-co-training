# -*- coding: utf-8 -*-
""" nns/mixins/checkpoints/standard_checkpoints """

import os
from collections import OrderedDict
from typing import List, Union

import torch
from torch.optim.optimizer import Optimizer
from logzero import logger

from nns.mixins.checkpoints.base import CheckPointBaseMixin


__all__ = ['DACheckPointMixin']


class DACheckPointMixin(CheckPointBaseMixin):
    """
    Provides methods to save and load checkpoints for inference or resume training when working
    with disagreement attention (two models trained together)

    Usage:
        class SomeClass(DACheckPointMixin):
            ...
    """

    best_single_model_checkpoint_pattern = 'best_chkpt_model_{}.pth.tar'
    merged_best_single_models_checkpoint_name = 'best_merged_single_models.pth.tar'

    def save_checkpoint(
            self, epoch: Union[int, float], optimizers: List[Optimizer], data_logger: dict,
            best_chkpt: bool = False, last_chkpt: bool = False
    ):
        """
        Saves the model as a checkpoint for inference and/or resuming training

        When best_chkpt = True, intrain_x_counter refers to the intrain_val_counter
        (see ModelMGRMixin.training method).

        When best_chkpt = False, intrain_x_counter refers to intrain_chkpt_counter
        (see ModelMGRMixin.training method). If epoch is an integer, we are not
        performing intrain validation (instead, we are saving data from the epoch
        evaluation); thus, intrain_x_counter is set to 0. On the other
        hand, when epoch is a float, we assign its integer part to epoch_ and its
        decimal part to intrain_x_counter.

        Kwargs:
            epoch           <int, float>: current epoch
            optimizers <List[Optimizer]>: List of optimizers objects
            data_logger           <dict>: dict with the tracked data (like lr, loss, metric, etc)
            best_chkpt            <bool>: If True the prefix 'best_' will be appended to the filename
            last_chkpt            <bool>: If True the prefix 'last_' will be appended to the filename
        """
        assert isinstance(epoch, (int, float)), type(epoch)
        assert epoch >= 0, f'{epoch}'
        assert isinstance(optimizers, list), type(optimizers)
        assert len(optimizers) == 2, 'optimizers must contains two instances of Optimizer'
        for optim in optimizers:
            assert isinstance(optim, Optimizer), type(optim)
        assert isinstance(data_logger, dict), type(data_logger)
        assert isinstance(best_chkpt, bool), type(best_chkpt)
        assert isinstance(last_chkpt, bool), type(last_chkpt)

        if isinstance(epoch, float):
            epoch_, intrain_x_counter = map(int, str(epoch).split('.'))
        else:
            epoch_ = epoch
            intrain_x_counter = 0

        # TODO: also save the scaler
        data = {
            'epoch': epoch_,
            'intrain_x_counter': intrain_x_counter,
            'model_state_dict1': self.module.model1.state_dict(),
            'model_state_dict2': self.module.model2.state_dict(),
            'optimizer1_state_dict': optimizers[0].state_dict(),
            'optimizer2_state_dict': optimizers[1].state_dict(),
            'data_logger': data_logger,
            'best_single_models': False  # flag to let us know how this checkpoint was built
        }

        if best_chkpt:
            filename = self.best_checkpoint_name
        elif last_chkpt:
            filename = self.last_checkpoint_name
        else:
            filename = self.checkpoint_pattern.format(f"_{epoch}")

        torch.save(data, os.path.join(self.dir_checkpoints, filename))

    def save_checkpoint_single_model(
            self, model: torch.nn.Module, epoch: int, optimizer: Optimizer, data_logger: dict, *,
            best_chkpt=False, last_chkpt=False, filename=''):
        """
        Saves the model as a checkpoint for inference and/or resuming training

        When best_chkpt = True, intrain_x_counter refers to the intrain_val_counter
        (see ModelMGRMixin.training method).

        When best_chkpt = False, intrain_x_counter refers to intrain_chkpt_counter
        (see ModelMGRMixin.training method). If epoch is an integer, we are not
        performing intrain validation (instead, we are saving data from the epoch
        evaluation); thus, intrain_x_counter is set to 0. On the other
        hand, when epoch is a float, we assign its integer part to epoch_ and its
        decimal part to intrain_x_counter.

        Kwargs:
            model    <torch.nn.Module>: model to be saved
            epoch         <int, float>: current epoch
            optimizer      <Optimizer>: optimizer instance
            data_logger         <dict>: dict with the tracked data (like lr, loss, metric, etc)
            best_chkpt          <bool>: If True the prefix 'best_' will be appended to the filename
            last_chkpt          <bool>: If True the prefix 'last_' will be appended to the filename
            filename             <str>: checkpoint filename
        """
        assert issubclass(model.__class__, torch.nn.Module), type(model)
        assert isinstance(epoch, (int, float)), type(epoch)
        assert epoch >= 0, f'{epoch}'
        assert isinstance(optimizer, Optimizer), type(optimizer)
        assert isinstance(data_logger, dict), type(data_logger)
        assert isinstance(best_chkpt, bool), type(best_chkpt)
        assert isinstance(last_chkpt, bool), type(last_chkpt)
        assert isinstance(filename, str), type(filename)

        if isinstance(epoch, float):
            epoch_, intrain_x_counter = map(int, str(epoch).split('.'))
        else:
            epoch_ = epoch
            intrain_x_counter = 0

        # TODO: also save the scaler
        data = {
            'epoch': epoch_,
            'intrain_x_counter': intrain_x_counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'data_logger': data_logger
        }

        if filename:
            pass
        elif best_chkpt:
            filename = self.best_checkpoint_name
        elif last_chkpt:
            filename = self.last_checkpoint_name
        else:
            filename = self.checkpoint_pattern.format(f"_{epoch}")

        torch.save(data, os.path.join(self.dir_checkpoints, filename))

    def merge_best_single_checkpoints(
            self, epoch: int, optimizers: List[Optimizer], data_logger: dict,
            model_idxs: List[int] = None
    ):
        """
        Loads the two best single model checkpoints and merges their data into a single checkpoint
        that can be used as the initial checkpoint fo DAModelMGR. Furthermore, the latest optimizers
        and data_logger are saved too so this checkpoint can also be used as initial checkpoint to
        continue the training process.

        kwargs:
            epoch                  <int>: training epoch
            optimizers <List[Optimizer]>: List of optimizers objects
            data_logger           <dict>: dict with the tracked data (like lr, loss, metric, etc)
            model_idxx       <List[int]>: List of integers containing the ids of the models.
                                          Default [1, 2]
        """
        assert isinstance(epoch, int), type(epoch)
        assert isinstance(optimizers, list), type(optimizers)
        assert len(optimizers) == 2, 'optimizers must contains two instances of Optimizer'
        for optim in optimizers:
            assert isinstance(optim, Optimizer), type(optim)
        assert isinstance(data_logger, dict), type(data_logger)
        if model_idxs:
            assert isinstance(model_idxs, list), type(model_idxs)
            assert len(model_idxs) == 2, 'model_idx must contain 2 numbers'
        else:
            model_idxs = [1, 2]

        chkpts = []

        for i in model_idxs:
            chkpt_path = os.path.join(
                self.dir_checkpoints, self.best_single_model_checkpoint_pattern.format(i))
            assert os.path.isfile(chkpt_path)

            chkpts.append(torch.load(chkpt_path))

        # TODO: find out if it's better to save the last optimizers or the optimizers
        #       from the very best models
        data = {
            'epoch': epoch,
            'intrain_x_counter': chkpts[0]['intrain_x_counter'],
            'model_state_dict1': chkpts[0]['model_state_dict'],
            'model_state_dict2': chkpts[1]['model_state_dict'],
            'optimizer1_state_dict': optimizers[0].state_dict(),
            'optimizer2_state_dict': optimizers[1].state_dict(),
            'data_logger': data_logger,
            'best_single_models': True  # flag to let us know how this checkpoint was built
        }

        torch.save(
            data,
            os.path.join(self.dir_checkpoints, self.merged_best_single_models_checkpoint_name)
        )

    def load_checkpoint(self, optimizers: List[Optimizer]):
        """
        Loads the checkpoint for inference and/or resuming training

        NOTE: The intrain_x_counter data is not considered when loading a checkpoint, because
        dataset elements can be shuffled making it impossible to continue the training
        in the exact position where the checkpoint was saved. Thus, we only consider the
        epoch number to load the data and continue with the training process. In this regard,
        when planning to continue the training process, it is recommended to set
        ini_checkpoint to any epoch checkpoint e.g. chkpt_<int>.pth.tar

        Kwargs:
            optimizers <List[Optimizer]>: list of optimizer instances

        Returns:
            current epoch (int), data_logger (dict)
        """
        assert isinstance(optimizers, list), type(optimizers)
        assert len(optimizers) == 2, 'optimizers must contain two optimizer instances'
        for optim in optimizers:
            assert isinstance(optim, Optimizer), type(optim)

        chkpt_path = os.path.join(self.dir_checkpoints, self.ini_checkpoint)
        assert os.path.isfile(chkpt_path), chkpt_path

        chkpt = torch.load(chkpt_path)
        self.module.model1.load_state_dict(chkpt['model_state_dict1'])
        self.module.model2.load_state_dict(chkpt['model_state_dict2'])
        optimizers[0].load_state_dict(chkpt['optimizer1_state_dict'])
        optimizers[1].load_state_dict(chkpt['optimizer2_state_dict'])

        # sending model and optimizers to the right device
        self.model.to(self.device)

        # TODO: also load the scaler
        for optimizer in optimizers:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        logger.debug(f'Checkpoint {chkpt_path} loaded.')

        return chkpt['epoch'], chkpt['data_logger']

    def get_best_checkpoint_data(self, best_single_models: bool = False):
        """
        Tries to load and return the best checkpoint saved. If it cannot, then None is returned

        Kwargs:
            best_single_models <bool>: if True it trieds to load and return the merged best single models
                                       checkpoint (it means your DAModelMGR has been executed with
                                       joint_values=False).
                                       Default False

        Returns:
            checkpoint <dict> or None
        """
        assert isinstance(best_single_models, bool), type(best_single_models)

        if best_single_models:
            chkpt_path = os.path.join(self.dir_checkpoints, self.merged_best_single_models_checkpoint_name)
        else:
            chkpt_path = os.path.join(self.dir_checkpoints, self.best_checkpoint_name)

        if os.path.isfile(chkpt_path):
            return torch.load(chkpt_path)

        return None

    def load(self, models: List[OrderedDict]):
        """
        Loads a list of state dicts models <OrderedDict> only for inference

        kwargs:
            models <List[OrderedDict]>: List of model state dicts
        """
        assert isinstance(models, list), type(models)
        assert len(models) == 2, 'Two model state dicts are required'

        for model in models:
            assert isinstance(model, OrderedDict), type(model)

        self.module.model1.load_state_dict(models[0])
        self.module.model2.load_state_dict(models[1])

        logger.debug('Model state dicts for model 1 and 2 loaded.')
