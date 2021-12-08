# -*- coding: utf-8 -*-
""" nns/callbacks/plotters/masks """

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class MaskPlotter:
    """
    Holds methods to plot and save masks

    Usage:
        mask_plotter = MaskPlotter(label_data=Label)

        for batch in dataloader:
           # some code
           mask_plotter(images, true_masks, pred_masks, filenames)
           # some code
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            labels_data <object>: class containing all the details of the classes. e.g.:
                Detail = namedtuple('Detail', ['colour', 'id', 'name', 'file_label', 'RGB'])

                class Label:
                    HEPATOCYTE = Detail('pink', 1, 'Hepatocyte', 'H', (238, 194, 204))
                    NECROSIS = Detail('c', 2, 'Necrosis', 'N', (111, 189, 190))
                    FIBROSIS = Detail('lightgrey', 3, 'Fibrosis', 'F', (211, 211, 211))
                    TUMOUR = Detail('saddlebrown', 4, 'Tumour', 'T', (123, 73, 22))
                    INFLAMMATION = Detail('g', 5, 'Inflammation', 'I', (144, 119, 3))
                    MUCIN = Detail('r', 6, 'Mucin', 'M', (217, 47, 19))
                    BLOOD = Detail('purple', 7, 'Blood', 'B', (108, 0, 128))
                    FOREIGN_BODY = Detail('royalblue', 8, 'Foreign body', 'D', (80, 94, 225))
                    MACROPHAGES = Detail('k', 9, 'Macrophages', 'C', (0, 0, 0))
                    BILE_DUCT = Detail('gold', 10, 'Bile Duct', 'Y', (244, 220, 5))

                    LABELS = (HEPATOCYTE, NECROSIS, FIBROSIS, TUMOUR, INFLAMMATION, MUCIN, BLOOD,
                              FOREIGN_BODY, MACROPHAGES, BILE_DUCT)
                    FILE_LABELS = tuple(label.file_label for label in LABELS)
                    CMAPS = tuple(
                        LinearSegmentedColormap.from_list(f'{label.name}_cmap', [(0, label.colour), (1, 'white')])
                        for label in LABELS
                    )

            mask_threshold <float>: minimum value to consider a predicted mask value as correct
            alpha       <float>: alpha channel value
            saving_dir    <str>: Directory where to save the plotted masks as PNG files.
                                 Default 'plotted_masks'
            dir_per_file <bool>: If true, the data will be saved in folders created
                                 using the filenames. Default False
            superimposed <bool>: Whether or not superimpose the masks at the original images
                                 Default False
            max_values   <bool>: If true, plots only superimposed non-decoupled masks with the highest
                                 score. Default False
            decoupled    <bool>: Whether or not save the masks in differente files per
                                 class/label. Default False
        """
        self.labels_data = kwargs.get('labels_data')
        self.mask_threshold = kwargs.get('mask_threshold', 0.5)
        self.alpha = kwargs.get('alpha', .7)
        self.saving_dir = kwargs.get('saving_dir', 'plotted_masks')
        self.dir_per_file = kwargs.get('dir_per_file', False)
        self.superimposed = kwargs.get('superimposed', False)
        self.max_values = kwargs.get('max_values', False)
        self.decoupled = kwargs.get('decoupled', False)

        assert isinstance(self.labels_data, object), type(self.labels_data)
        assert isinstance(self.mask_threshold, float), type(self.mask_threshold)
        assert 0 < self.mask_threshold < 1, self.mask_threshold
        assert isinstance(self.alpha, float), type(self.alpha)
        assert isinstance(self.saving_dir, str), type(self.saving_dir)
        assert isinstance(self.dir_per_file, bool), type(self.dir_per_file)
        assert isinstance(self.superimposed, bool), type(self.superimposed)
        assert isinstance(self.max_values, bool), type(self.max_values)
        assert isinstance(self.decoupled, bool), type(self.decoupled)

    def __call__(self, *args):
        """ functor call """
        self.plot(*args)

    def plot_images(self, images, filenames):
        """
        Plots the images and saves them as PNG files

        Args:
            images <torch.Tensor>: prediction with shape [batch, channels, height, width]
            filenames     <tuple>: tuple of image names

        """
        assert isinstance(images, torch.Tensor), type(images)
        assert isinstance(filenames, tuple)

        for image, filename in zip(images, filenames):
            saving_dir_ = os.path.join(self.saving_dir, filename) if self.dir_per_file else self.saving_dir

            if not os.path.isdir(saving_dir_):
                os.makedirs(saving_dir_)

            plt.imsave(
                os.path.join(saving_dir_, f'{filename}.ann.png'),
                (image.permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
            )

    def plot_masks(self, **kwargs):
        """
        Plots the masks and saves them as PNG files

        Kwargs:
            masks      <torch.Tensor>: masks with shape [batch, classes, height, width]
            filenames         <tuple>: tuple of masks names (any name to be used as filename)
            suffix              <str>: suffix used when saving the image. Default ''
        """
        masks = kwargs.get('masks')
        filenames = kwargs.get('filenames')
        suffix = kwargs.get('suffix', '')

        assert isinstance(masks, torch.Tensor), type(masks)
        assert isinstance(filenames, tuple), type(filenames)
        assert isinstance(suffix, str), type(suffix)

        # adding an extra class full of zeros to represent anything else than the
        # defined classes like background or any other not identified thing
        preds_plus_bg = torch.cat(
            [torch.zeros((masks.shape[0], 1, *masks.shape[2:])), masks.cpu()], dim=1)
        preds_plus_bg[preds_plus_bg <= self.mask_threshold] = 0
        preds_plus_bg = torch.argmax(preds_plus_bg, dim=1)

        for pred, filename in zip(preds_plus_bg, filenames):
            saving_dir_ = os.path.join(self.saving_dir, filename) if self.dir_per_file else self.saving_dir

            if not os.path.isdir(saving_dir_):
                os.makedirs(saving_dir_)

            rgb = torch.full((3, *pred.size()), 255)

            for label in self.labels_data.LABELS:
                colour_idx = pred == label.id
                rgb[0][colour_idx] = label.RGB[0]
                rgb[1][colour_idx] = label.RGB[1]
                rgb[2][colour_idx] = label.RGB[2]

            plt.imsave(
                f'{os.path.join(saving_dir_, filename)}.mask{suffix}.png',
                rgb.permute(1, 2, 0).numpy().astype('uint8')
            )

    def plot_superimposed_masks(self, **kwargs):
        """
        Plots the masks and saves them as PNG files

        Kwargs:
            images <torch.Tensor>: images with shape [batch, channels, height, width]
            masks  <torch.Tensor>: masks with shape [batch, classes, height, width]
            filenames     <tuple>: tuple of prediction names
            suffix          <str>: suffix used when saving the image. Default ''
        """
        images = kwargs.get('images')
        masks = kwargs.get('masks')
        filenames = kwargs.get('filenames')
        suffix = kwargs.get('suffix', '')

        assert isinstance(images, torch.Tensor), type(images)
        assert isinstance(masks, torch.Tensor), type(masks)
        assert isinstance(filenames, tuple), type(filenames)
        assert isinstance(suffix, str), type(suffix)

        for img, mask, filename in zip(images, masks, filenames):
            saving_dir_ = os.path.join(self.saving_dir, filename) if self.dir_per_file else self.saving_dir

            if not os.path.isdir(saving_dir_):
                os.makedirs(saving_dir_)

            fig = plt.figure()
            fig.set_size_inches(1. * img.shape[1] / img.shape[2], 1, forward=False)
            axes = plt.Axes(fig, [0., 0., 1., 1.])
            axes.set_axis_off()
            fig.add_axes(axes)
            axes.imshow((img.permute(1, 2, 0).cpu().numpy()*255).astype('uint8'))

            for label, cmap in zip(self.labels_data.LABELS, self.labels_data.CMAPS):
                mask_ = (mask[label.id-1] > self.mask_threshold).cpu().numpy().astype('uint8')
                unique_values = np.unique(mask_)

                if unique_values.shape[0] == 1 and unique_values[0] == 0:
                    continue

                masked = np.ma.masked_where(mask_ == 0, mask_)
                axes.imshow(masked, cmap, alpha=self.alpha)

            plt.savefig(
                f'{os.path.join(saving_dir_, filename)}.mask{suffix}.png', dpi=img.shape[1]
            )

            plt.clf()
            plt.close()

    def plot_superimposed_masks_with_max_values(self, **kwargs):
        """
        Plots the masks using only the highest value per pixel and saves them as PNG files

        Kwargs:
            images <torch.Tensor>: images with shape [batch, channels, height, width]
            masks  <torch.Tensor>: masks with shape [batch, classes, height, width]
            filenames     <tuple>: tuple of prediction names
            suffix          <str>: suffix used when saving the image. Default ''
        """
        images = kwargs.get('images')
        masks = kwargs.get('masks')
        filenames = kwargs.get('filenames')
        suffix = kwargs.get('suffix', '')

        assert isinstance(images, torch.Tensor), type(images)
        assert isinstance(masks, torch.Tensor), type(masks)
        assert isinstance(filenames, tuple), type(filenames)
        assert isinstance(suffix, str), type(suffix)

        # adding an extra class full of zeros to represent anything else than the
        # defined classes like background or any other not identified thing
        preds_plus_bg = torch.cat(
            [torch.zeros((masks.shape[0], 1, *masks.shape[2:])), masks.cpu()], dim=1)
        preds_plus_bg[preds_plus_bg <= self.mask_threshold] = 0
        preds_plus_bg = torch.argmax(preds_plus_bg, dim=1)

        for img, mask, filename in zip(images, preds_plus_bg, filenames):
            saving_dir_ = os.path.join(self.saving_dir, filename) if self.dir_per_file else self.saving_dir

            if not os.path.isdir(saving_dir_):
                os.makedirs(saving_dir_)

            fig = plt.figure()
            fig.set_size_inches(1. * img.shape[1] / img.shape[2], 1, forward=False)
            axes = plt.Axes(fig, [0., 0., 1., 1.])
            axes.set_axis_off()
            fig.add_axes(axes)
            axes.imshow((img.permute(1, 2, 0).cpu().numpy()*255).astype('uint8'))
            mask_ = mask.cpu().numpy().astype('uint8')
            unique_values = np.unique(mask_)

            for label, cmap in zip(self.labels_data.LABELS, self.labels_data.CMAPS):
                if label.id not in unique_values:
                    continue

                masked = np.ma.masked_where(mask_ != label.id, mask_)
                axes.imshow(masked, cmap, alpha=self.alpha)

            plt.savefig(
                f'{os.path.join(saving_dir_, filename)}.mask{suffix}.png', dpi=img.shape[1]
            )

            plt.clf()
            plt.close()

    def plot_superimposed_decoupled_masks(self, **kwargs):
        """
        Separately plots the masks and saves them in label-based PNG files

        Kwargs:
            images <torch.Tensor>: images with shape [batch, channels, height, width]
            masks  <torch.Tensor>: masks with shape [batch, classes, height, width]
            filenames     <tuple>: tuple of prediction names
            suffix          <str>: suffix used when saving the image. Default ''

        """
        images = kwargs.get('images')
        masks = kwargs.get('masks')
        filenames = kwargs.get('filenames')
        suffix = kwargs.get('suffix', '')

        assert isinstance(images, torch.Tensor), type(images)
        assert isinstance(masks, torch.Tensor), type(masks)
        assert isinstance(filenames, tuple), type(filenames)
        assert isinstance(suffix, str), type(suffix)

        for img, mask, filename in zip(images, masks, filenames):
            saving_dir_ = os.path.join(self.saving_dir, filename) if self.dir_per_file else self.saving_dir

            if not os.path.isdir(saving_dir_):
                os.makedirs(saving_dir_)

            for label, cmap in zip(self.labels_data.LABELS, self.labels_data.CMAPS):
                mask_ = (mask[label.id-1] > self.mask_threshold).cpu().numpy().astype('uint8')
                unique_values = np.unique(mask_)

                if unique_values.shape[0] == 1 and unique_values[0] == 0:
                    continue

                fig = plt.figure()
                fig.set_size_inches(1. * img.shape[1] / img.shape[2], 1, forward=False)
                axes = plt.Axes(fig, [0., 0., 1., 1.])
                axes.set_axis_off()
                fig.add_axes(axes)
                axes.imshow((img.permute(1, 2, 0).cpu().numpy()*255).astype('uint8'))
                masked = np.ma.masked_where(mask_ == 0, mask_)
                axes.imshow(masked, cmap, alpha=self.alpha)
                plt.savefig(
                    f'{os.path.join(saving_dir_, filename)}_{label.file_label}.mask{suffix}.png',
                    dpi=img.shape[1]
                )
                plt.clf()
                plt.close()

    def plot(self, images, true_masks, pred_masks, filenames):
        """
        Plots the masks based on the configuration of the instance

        Args:
            images     <torch.Tensor>: images with shape [batch, channels, height, width]
            true_masks <torch.Tensor>: ground truth masks with shape [batch, classes, height, width]
            masks      <torch.Tensor>: predicted masks with shape [batch, classes, height, width]
            filenames         <tuple>: tuple of prediction names (filenames for the png files)
        """
        self.plot_images(images, filenames)

        if not self.superimposed:
            self.plot_masks(masks=true_masks, filenames=filenames)
            self.plot_masks(masks=pred_masks, filenames=filenames, suffix='.pred')
        else:
            if self.decoupled:
                self.plot_superimposed_decoupled_masks(
                    images=images, masks=true_masks, filenames=filenames)
                self.plot_superimposed_decoupled_masks(
                    images=images, masks=pred_masks, filenames=filenames, suffix='.pred')
            else:
                if self.max_values:
                    self.plot_superimposed_masks_with_max_values(
                        images=images, masks=true_masks, filenames=filenames)
                    self.plot_superimposed_masks_with_max_values(
                        images=images, masks=pred_masks, filenames=filenames, suffix='.pred')
                else:
                    self.plot_superimposed_masks(
                        images=images, masks=true_masks, filenames=filenames)
                    self.plot_superimposed_masks(
                        images=images, masks=pred_masks, filenames=filenames, suffix='.pred')
