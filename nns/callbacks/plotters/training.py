# -*- coding: utf-8 -*-
""" nns/callbacks/plotters/training """

import os

import matplotlib.pyplot as plt
import numpy as np


class TrainingPlotter:
    """
    Holds methods to plot and save training loss, metric and learning rate

    Usage:
        TrainingPlotter(
            train_loss=data_logger['train_loss'],
            train_metric=data_logger['train_metric'],
            val_loss=data_logger['val_loss'],
            val_metric=data_logger['val_metric'],
            lr=data_logger['lr'],
            clip_interval=(0, 1),
        )(save=True, show=False, xlabel='Sets of X images')
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            train_loss   <Optional[list]>: list containing the training loss logs
            train_metric <Optional[list]>: list containing the training metric logs
            val_loss     <Optional[list]>: list containing the validation loss logs
            val_metric   <Optional[list]>: list containing the validation metric logs
            lr           <Optional[list]>: list containing the training learning rate logs
            clip_interval <Optional[Union[list, tuple]]>: Range [min, max] used to clamp all elements
        """
        self.train_loss = kwargs.get('train_loss', [])
        self.train_metric = kwargs.get('train_metric', [])
        self.val_loss = kwargs.get('val_loss', [])
        self.val_metric = kwargs.get('val_metric', [])
        self.lr = kwargs.get('lr', [])
        self.clip_interval = kwargs.get('clip_interval', None)

        assert isinstance(self.train_loss, list), type(self.train_loss)
        assert isinstance(self.train_metric, list), type(self.train_metric)
        assert isinstance(self.val_loss, list), type(self.val_loss)
        assert isinstance(self.val_metric, list), type(self.val_metric)
        assert isinstance(self.lr, list), type(self.lr)

        self.train_loss = np.array(self.train_loss)
        self.train_metric = np.array(self.train_metric)
        self.val_loss = np.array(self.val_loss)
        self.val_metric = np.array(self.val_metric)
        self.lr = np.array(self.lr)

        if self.clip_interval is not None:
            assert isinstance(self.clip_interval, (list, tuple)), type(self.clip_interval)
            self.train_loss = self.train_loss.clip(*self.clip_interval)
            self.train_metric = self.train_metric.clip(*self.clip_interval)
            self.val_loss = self.val_loss.clip(*self.clip_interval)
            self.val_metric = self.val_metric.clip(*self.clip_interval)

    def __call__(self, **kwargs):
        """ functor call """
        return self.plot_all(**kwargs)

    def plot_losses_and_metrics(self, **kwargs):
        """
        Plots the losses and metrics

        Kwargs:
            title           <str>: plot title. Default ''
            xlabel          <str>: Label for X axis. Default 'Epochs'
            ylabel          <str>: Label for Y axis. Default 'Loss & Metric'
            legend_kwargs  <dict>: Dictionary contaning data for the legend method.
                                   Default dict(shadow=True, fontsize=8, bbox_to_anchor=(1.1, 1.15), loc='upper right'))
                                   Other good option: dict(shadow=True, fontsize=8, loc='best')
            plot_kwargs    <dict>: Dictionary contaning data for the plot method.
                                   Default {'linewidth': 1.5}
            save           <bool>: Whether or not save to disk. Default False
            dpi <float, 'figure'>: The resolution in dots per inch.
                                   If 'figure', use the figure's dpi value. For high quality images
                                   set it to 300.
                                   Default 'figure'
            show           <bool>: Where or not display the image. Default True
            saving_path     <str>: Full path to the image file to save the image.
                                   Default 'losses_metrics.png'
            train_loss_label <str>: Label for the train_loss line
            val_loss_label  <str>: Label for the val_loss line
            train_metric_label <str>: Label for the train_metric line
            val_metric_label <str>: Label for the val_metric line
            xticks_labels <Union[list, np.ndarray>]: The labels for xticks locations. Default None
        """
        title = kwargs.get('title', '')
        xlabel = kwargs.get('xlabel', 'Epochs')
        ylabel = kwargs.get('ylabel', 'Loss & Metric')
        legend_kwargs = kwargs.get(
            'legend_kwargs', dict(shadow=True, fontsize=8, bbox_to_anchor=(1.1, 1.15), loc='upper right'))
        plot_kwargs = kwargs.get('plot_kwargs', {'linewidth': 1.5})
        save = kwargs.get('save', False)
        dpi = kwargs.get('dpi', 'figure')
        show = kwargs.get('show', True)
        saving_path = kwargs.get('saving_path', 'losses_metrics.png')
        train_loss_label = kwargs.get('train_loss_label', 'train loss')
        val_loss_label = kwargs.get('val_loss_label', 'val loss')
        train_metric_label = kwargs.get('train_metric_label', 'train metric')
        val_metric_label = kwargs.get('val_metric_label', 'val metric')
        xticks_labels = kwargs.get('xticks_labels', None)

        assert isinstance(title, str), type(title)
        assert isinstance(xlabel, str), type(xlabel)
        assert isinstance(ylabel, str), type(ylabel)
        assert isinstance(legend_kwargs, dict), type(legend_kwargs)
        assert isinstance(plot_kwargs, dict), type(plot_kwargs)
        assert isinstance(save, bool), type(save)
        assert isinstance(dpi, float) or dpi == 'figure'
        assert isinstance(show, bool), type(show)
        assert isinstance(saving_path, str), type(saving_path)
        assert isinstance(train_loss_label, str), type(train_loss_label)
        assert isinstance(val_loss_label, str), type(val_loss_label)
        assert isinstance(train_metric_label, str), type(train_metric_label)
        assert isinstance(val_metric_label, str), type(val_metric_label)
        if xticks_labels is not None:
            assert isinstance(xticks_labels, (list, np.ndarray)), type(xticks_labels)

        dirname = os.path.dirname(saving_path)

        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        x_data = np.arange(0, len(self.train_loss)) if self.train_loss.size else \
            np.arange(0, len(self.train_metric))

        # plt.clf()
        plt.style.use("ggplot")
        plt.figure()
        legend_handles = []

        if self.train_loss.size:
            line1, = plt.plot(x_data, self.train_loss, label=train_loss_label, **plot_kwargs)
            legend_handles.append(line1)

        if self.val_loss.size:
            line2, = plt.plot(x_data, self.val_loss, label=val_loss_label, **plot_kwargs)
            legend_handles.append(line2)

        if self.train_metric.size:
            line3, = plt.plot(x_data, self.train_metric, label=train_metric_label, **plot_kwargs)
            legend_handles.append(line3)

        if self.val_metric.size:
            line4,  = plt.plot(x_data, self.val_metric, label=val_metric_label, **plot_kwargs)
            legend_handles.append(line4)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(handles=legend_handles, **legend_kwargs)
        plt.xticks(x_data, xticks_labels)

        if save:
            plt.savefig(saving_path, dpi=dpi)

        if show:
            plt.show()

        plt.close()

    def plot_learning_rate(self, **kwargs):
        """
        Plots the learning rate

        Kwargs:
            title           <str>: plot title. Default 'Learning Rate'
            xlabel          <str>: Label for X axis. Default 'Epochs'
            ylabel          <str>: Label for Y axis. Default 'Learning Rate'
            xticks_labels <Union[list, np.ndarray>]: The labels for xticks locations. Default None
            plot_kwargs    <dict>: Dictionary contaning data for the plot method.
                                   Default {'linewidth': 1.5}
            save           <bool>: Whether or not save to disk. Default False
            dpi <float, 'figure'>: The resolution in dots per inch.
                                   If 'figure', use the figure's dpi value. For high quality images
                                   set it to 300.
                                   Default 'figure'
            show           <bool>: Where or not display the image. Default True
            saving_path     <str>: Full path to the image file to save the image.
                                   Default 'learning_rate.png'
        """
        if len(self.lr) == 0:
            return

        title = kwargs.get('title', 'Learning Rate')
        xlabel = kwargs.get('xlabel', 'Epochs')
        ylabel = kwargs.get('ylabel', '')
        xticks_labels = kwargs.get('xticks_labels', None)
        plot_kwargs = kwargs.get('plot_kwargs', {'linewidth': 1.5})
        save = kwargs.get('save', False)
        dpi = kwargs.get('dpi', 'figure')
        show = kwargs.get('show', True)
        saving_path = kwargs.get('saving_path', 'learning_rate.png')

        assert isinstance(title, str), type(title)
        assert isinstance(xlabel, str), type(xlabel)
        assert isinstance(ylabel, str), type(ylabel)
        if xticks_labels is not None:
            assert isinstance(xticks_labels, (list, np.ndarray)), type(xticks_labels)
        assert isinstance(plot_kwargs, dict), type(plot_kwargs)
        assert isinstance(save, bool), type(save)
        assert isinstance(dpi, float) or dpi == 'figure'
        assert isinstance(show, bool), type(show)
        assert isinstance(saving_path, str), type(saving_path)

        dirname = os.path.dirname(saving_path)

        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        x_data = np.arange(0, len(self.lr))

        # plt.clf()
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(x_data, self.lr, **plot_kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x_data, xticks_labels)

        if save:
            plt.savefig(saving_path, dpi=dpi)

        if show:
            plt.show()

        plt.close()

    def plot_all(self, **kwargs):
        """
        Tries to plot the losses, metrics and learning rate

        Kwargs:
            lm_title          <str>: plot title for loss and metric plot. Default ''
            lr_title          <str>: plot title for learning rate plot. Default 'Learning Rate'
            lm_ylabel         <str>: Label for Y axis from loss and metrics plot.
                                     Default 'Loss & Metric'
            lr_ylabel         <str>: Label for Y axis from learning rate plot.
                                     Default 'Learning Rate'
            lm_xticks_labels <Union[list, np.ndarray>]: The labels for xticks locations of loss and metrics
                                     plots. Default None
            lr_xticks_labels <Union[list, np.ndarray>]: The labels for xticks locations of learning rate
                                     plot. Default None
            lm_legend_kwargs <dict>: Dictionary contaning data for the legend method from loss
                                     and metrics plot.
                                     Default dict(shadow=True, fontsize=8, bbox_to_anchor=(1.1, 1.15), loc='upper right'))
                                     Other good option: dict(shadow=True, fontsize=8, loc='best')
            lm_saving_path    <str>: Full path to the image file to save the loss and metric plot.
                                     Default 'losses_metrics.png'
            lr_saving_path       <str>: Full path to the image file to save the learning rate plot.
                                     Default 'learning_rate.png'
        """
        lm_title = kwargs.pop('lm_title', '')
        lr_title = kwargs.pop('lr_title', 'Learning Rate')
        lm_ylabel = kwargs.pop('lm_ylabel', 'Loss & Metric')
        lr_ylabel = kwargs.pop('lr_ylabel', '')
        lm_xticks_labels = kwargs.pop('lm_xticks_labels', None)
        lr_xticks_labels = kwargs.pop('lr_xticks_labels', None)

        lm_legend_kwargs = kwargs.get(
            'lm_legend_kwargs', dict(shadow=True, fontsize=8, bbox_to_anchor=(1.1, 1.15), loc='upper right'))

        lm_saving_path = kwargs.get('lm_saving_path', 'losses_metrics.png')
        lr_saving_path = kwargs.get('lr_saving_path', 'learning_rate.png')

        if len(self.train_loss) or len(self.train_metric) or len(self.val_loss) or \
           len(self.val_metric):
            self.plot_losses_and_metrics(
                title=lm_title,
                ylabel=lm_ylabel,
                legend_kwargs=lm_legend_kwargs,
                saving_path=lm_saving_path,
                xticks_labels=lm_xticks_labels,
                **kwargs
            )

        if len(self.lr):
            self.plot_learning_rate(
                title=lr_title,
                ylabel=lr_ylabel,
                saving_path=lr_saving_path,
                xticks_labels=lr_xticks_labels,
                **kwargs
            )
