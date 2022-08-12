# -*- coding: utf-8 -*-
""" nns/mixins/checkpoints/modular_unet4plus_checkpoints """

import os
from typing import Union, Tuple

import torch
from logzero import logger
from torch.optim.optimizer import Optimizer

from nns.mixins.checkpoints.base import CheckPointBaseMixin


__all__ = ['SeveralOptimizersCheckPointMixin']


class SeveralOptimizersCheckPointMixin(CheckPointBaseMixin):
    """
    Provides standard methods to save and load checkpoints for inference or resume training for
    the Neuronal Networks with several optimizers

    Usage:
        class SomeClass(SeveralOptimizersCheckPointMixin):
            ...
    """

    def save_checkpoint(
            self, epoch: Union[int, float], optimizers: Tuple[Optimizer], data_logger: dict, best_chkpt: bool = False,
            last_chkpt: bool = False
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
            epoch         <int, float>: current epoch
            optimizers <Tuple[Optimizer]>: List of optimizer instances
            data_logger         <dict>: dict with the tracked data (like lr, loss, metric, etc)
            best_chkpt          <bool>: If True the prefix 'best_' will be appended to the filename
            last_chkpt          <bool>: If True the prefix 'last_' will be appended to the filename
        """
        assert isinstance(epoch, (int, float)), type(epoch)
        assert epoch >= 0, f'{epoch}'
        assert isinstance(optimizers, tuple), type(optimizers)
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dicts': [optim.state_dict() for optim in optimizers],
            'data_logger': data_logger
        }

        if best_chkpt:
            filename = self.best_checkpoint_name
        elif last_chkpt:
            filename = self.last_checkpoint_name
        else:
            filename = self.checkpoint_pattern.format(f"_{epoch}")

        torch.save(data, os.path.join(self.dir_checkpoints, filename))

    def load_checkpoint(self, optimizers: Tuple[Optimizer]):
        """
        Loads the checkpoint for inference and/or resuming training

        NOTE: The intrain_x_counter data is not considered when loading a checkpoint, because
        dataset elements can be shuffled making it impossible to continue the training
        in the exact position where the checkpoint was saved. Thus, we only consider the
        epoch number to load the data and continue with the training process. In this regard,
        when planning to continue the training process, it is recommended to set
        ini_checkpoint to any epoch checkpoint e.g. chkpt_<int>.pth.tar

        Kwargs:
            optimizers <Tuple[Optimizer]>: list of optimizer instances

        Returns:
            current epoch (int), data_logger (dict)
        """
        assert isinstance(optimizers, tuple), type(optimizers)

        chkpt_path = os.path.join(self.dir_checkpoints, self.ini_checkpoint)
        assert os.path.isfile(chkpt_path), chkpt_path

        chkpt = torch.load(chkpt_path)
        self.model.load_state_dict(chkpt['model_state_dict'])

        for idx in range(len(optimizers)):
            optimizers[idx].load_state_dict(chkpt['optimizer_state_dicts'][idx])

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
