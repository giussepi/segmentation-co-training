# -*- coding: utf-8 -*-
""" nns/mixins/checkpoints """

import os

import torch
from logzero import logger


class CheckPointMixin:
    """
    Provides methods to save and load checkpoints for inference or resume training

    Usage:
        class SomeClass(CheckPointMixin):
            ...
    """

    checkpoint_pattern = 'chkpt_{}.pth.tar'

    def save_checkpoint(self, epoch, optimizer, data_logger):
        """
        Saves the model as a checkpoint for inference and/or resuming training
        Args:
            epoch                <int, float>: current epoch
            optimizer <self.optimizer>: optimizer instance
            data_logger         <dict>: dict with the tracked data (like lr, loss, metric, etc)
        """
        assert isinstance(epoch, (int, float))
        assert epoch >= 0
        assert isinstance(optimizer, self.optimizer)
        assert isinstance(data_logger, dict)

        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'data_logger': data_logger
        }
        torch.save(
            data,
            os.path.join(
                self.dir_checkpoints,
                self.checkpoint_pattern.format(data['epoch'])
            )
        )

    def load_checkpoint(self, optimizer):
        """
        Loads the checkpoint for inference and/or resuming training

        Args:
            optimizer (self.optimizer): optimizer instance

        Returns:
            current epoch (int), data_logger (dict)
        """
        assert isinstance(optimizer, self.optimizer)

        chkpt_path = os.path.join(self.dir_checkpoints, self.ini_checkpoint)
        assert os.path.isfile(chkpt_path), chkpt_path

        chkpt = torch.load(chkpt_path)
        self.model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])

        # sending model and optimizer to the right device
        self.model.to(self.device)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        logger.info(f'Checkpoint {chkpt_path} loaded.')

        return chkpt['epoch'], chkpt['data_logger']

    def save(self, filename='best_model.pth'):
        """
        Saves the model only for inference

        Args:
            filename <str>: file name to be used to save the model

        """
        assert isinstance(filename, str), type(filename)

        torch.save(self.model.state_dict(), os.path.join(self.dir_checkpoints, filename))

    def load(self, filename='best_model.pth'):
        """
        Loads the model only for inference

        Args:
            filename <str>: file name to be loaded
        """
        assert isinstance(filename, str), type(filename)

        file_path = os.path.join(self.dir_checkpoints, filename)

        assert os.path.isfile(file_path), file_path

        self.model.load_state_dict(torch.load(file_path))

        logger.info(f'Model {file_path} loaded.')
