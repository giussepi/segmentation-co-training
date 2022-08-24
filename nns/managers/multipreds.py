# -*- coding: utf-8 -*-
""" nns/managers/multipreds """

from unittest.mock import MagicMock

import torch
from gtorch_utils.nns.managers.exceptions import ModelMGRImageChannelsError
from logzero import logger
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from settings import USE_AMP
from nns.callbacks.plotters.masks import MaskPlotter
from nns.mixins.managers import MultiPredsModelMGRMixin


__all__ = ['MultiPredsModelMGR']


class MultiPredsModelMGR(MultiPredsModelMGRMixin):
    """
    General segmentation manager class for models that returns multiple masks/logits and use a single
    optimizer. All the losses from the masks are backpropagated individually; then, the model weights are
    updated using the unique optimizer.

    Note: The model passed to the manager must have a module_names attribute containing the some names
    representing the output masks. Thus, if the model returns 4 prediction masks, then its module_names
    attribute should have 4 names e.g. self.module_names = ['de1', 'de2', 'd3', 'd4']

    """

    def get_validation_data(self, batch):
        """
        Returns the data to be used for the validation or test

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            imgs<torch.Tensor>, true_masks<torch.Tensor>, masks_pred<tuple of torch.Tensors>, labels<list>,
            label_names<list>, num_crops <int>
        """
        assert isinstance(batch, dict)
        assert len(batch) > 0, 'the provided batch is empty'

        imgs = batch['image']
        true_masks = batch['mask']
        labels = batch.get('label', ['']*self.testval_dataloader_kwargs['batch_size'])
        label_names = batch.get('label_name', ['']*self.testval_dataloader_kwargs['batch_size'])

        # commenting out main label validation because at level 1
        # while creating the crops with the desired size some of them
        # could not have data in the main label
        # for i in range(labels.shape[0]):
        #     assert true_masks[i][labels[i]].max() == 1, labels[i].item()

        num_crops = 1

        # TODO: see how to use and process label_names
        imgs, labels, true_masks, _ = self.reshape_data(imgs, labels, true_masks)

        if imgs.shape[1] != self.module.n_channels:
            raise ModelMGRImageChannelsError(self.module.n_channels, imgs.shape[1])

        imgs = imgs.to(device=self.device, dtype=torch.float32)
        # changing this becaue of the error
        # RuntimeError: _thnn_conv_depthwise2d_forward not supported on CUDAType for Long
        # mask_type = torch.float32 if self.module.n_classes == 1 else torch.long
        # mask_type = torch.float32
        true_masks = true_masks.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            masks_pred = self.model(imgs)

        assert isinstance(masks_pred, (list, tuple)), type(masks_pred)

        return imgs, true_masks, masks_pred, labels, label_names, num_crops

    def validation_step(self, **kwargs):
        """
        Logic to perform the validation step per batch

        Kwargs:
            batch       <dict>: Dictionary contaning batch data
            testing     <bool>: Whether it is performing testing or validation.
                                Default False
            plot_to_png <bool>: Whether or not save the predictions as PNG files. This option
                                is useful to visually examine the predicted masks.
                                Default False
            mask_plotter <MaskPlotter, None>: Optional MaskPlotter instance.
                                Default None
            imgs_counter <int>: Counter of processed images. Default 0
            apply_threshold <bool>: Whether or not apply thresholding to the predicted mask.
                                Default True

        Returns:
            losses<List[torch.Tensor]>, extra_data<dict>
        """
        batch = kwargs.get('batch')
        testing = kwargs.get('testing', False)
        plot_to_png = kwargs.get('plot_to_png', False)
        mask_plotter = kwargs.get('mask_plotter', None)
        imgs_counter = kwargs.get('imgs_counter', 0)
        apply_threshold = kwargs.get('apply_threshold', True)

        assert isinstance(batch, dict), type(batch)
        assert isinstance(testing, bool), type(testing)
        assert isinstance(plot_to_png, bool), type(plot_to_png)
        if mask_plotter:
            assert isinstance(mask_plotter, MaskPlotter), type(mask_plotter)
        assert isinstance(imgs_counter, int), type(imgs_counter)
        assert isinstance(apply_threshold, bool), type(apply_threshold)

        losses = []
        interpolate_mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            imgs, true_masks, masks_pred, labels, label_names, num_crops = self.get_validation_data(batch)

            if not testing:
                for idx, masks in enumerate(masks_pred):
                    if masks.shape != true_masks.shape:
                        true_masks_ = (F.interpolate(
                            true_masks, size=masks.size()[2:], mode=interpolate_mode, align_corners=False
                        ) >= self.mask_threshold).float()
                    else:
                        true_masks_ = true_masks

                    # applying lenient masks to first decoder output
                    if idx == 0:
                        losses.append(self.calculate_loss(self.criterions, masks, true_masks_*.85))
                    else:
                        losses.append(self.calculate_loss(self.criterions, masks, true_masks_))

                    # IMPORTANT NOTE:
                    # when using online data augmentation, it can return X crops instead of 1, so
                    # we need to modify this to loss / (n_val*X)
                    # because the loss is result of processing X times more crops than
                    # normal, so to properly calculate the final loss we need to divide it by
                    # number of batches times X. Here we only divide it by X, the final summation
                    # will be divided by num_baches at the validation method implementation
                    losses[-1] /= num_crops

        for idx, pred in enumerate(masks_pred, start=1):
            pred = torch.sigmoid(pred) if self.module.n_classes == 1 else torch.softmax(pred, dim=1)

            if testing and plot_to_png:
                # TODO: review if the logic of imgs_counter still works
                filenames = tuple(f'{imgs_counter + i}_{idx}' for i in range(1, pred.shape[0]+1))
                mask_plotter(imgs, true_masks, pred, filenames)

            if apply_threshold:
                # FIXME try calculating the metric without the threshold
                pred = (pred > self.mask_threshold).float()

            getattr(self, f'valid_metrics{idx}').update(pred, true_masks)

        extra_data = dict(
            imgs=imgs.detach().cpu(), pred=pred.detach().cpu(), true_masks=true_masks.detach().cpu(),
            labels=labels, label_names=label_names
        )

        return losses, extra_data

    def validation_post(self, **kwargs):
        """
        Actions to be performed after a validation step

        Kwargs:
            pred        <torch.Tensor>: predicted masks
            true_masks  <torch.Tensor>: ground truth masks
            labels              <list>: image labels
            imgs        <torch.Tensor>: batch of images
            label_names         <list>: label names
            writer <SummaryWriter, MagicMock>: instance of SummaryWriter or MagicMock
            validation_step      <int>: number of times the validation has been run
            global_step          <int>: global step counter
            val_extra_data      <dict>: dictionary containig the val_extra_data returned by the
                                        validation method. Default None
        """
        pred = kwargs.get('pred')
        true_masks = kwargs.get('true_masks')
        labels = kwargs.get('labels')  # not used in CoNSeP Binary
        imgs = kwargs.get('imgs')  # not used in CoNSeP Binary
        label_names = kwargs.get('label_names')
        writer = kwargs.get('writer')
        validation_step = kwargs.get('validation_step')
        global_step = kwargs.get('global_step')
        val_extra_data = kwargs.get('val_extra_data', None)

        assert isinstance(pred, torch.Tensor), type(pred)
        assert isinstance(true_masks, torch.Tensor), type(true_masks)
        assert isinstance(labels, list), type(labels)
        assert isinstance(imgs, torch.Tensor), type(imgs)
        assert isinstance(label_names, list), type(label_names)
        assert isinstance(writer, (SummaryWriter, MagicMock)), type(writer)
        assert isinstance(validation_step, int), type(validation_step)
        assert isinstance(global_step, int), type(global_step)
        if val_extra_data:
            assert isinstance(val_extra_data, dict), type(val_extra_data)

        # # plotting current training batch
        # for i, data in enumerate(zip(pred, true_masks, imgs)):
        #     writer.add_images(
        #         f'Training_{validation_step}_ImgBatch[{i}]/img', torch.unsqueeze(data[2], 0), global_step)
        #     writer.add_images(
        #         f'Training_{validation_step}_ImgBatch[{i}]/_gt',
        #         torch.unsqueeze(torch.unsqueeze(data[1][0, :, :], 0), 0),
        #         global_step
        #     )
        #     writer.add_images(
        #         f'Training_{validation_step}_ImgBatch[{i}]/_pred',
        #         torch.unsqueeze(torch.unsqueeze(data[0][0, :, :], 0), 0),
        #         global_step
        #     )

        # plotting last validation batch
        if val_extra_data:
            if self.data_dimensions == 3:
                # TODO: implement 3D data plotting to TensorBoardX
                logger.error('adding 3D data to tensorboardX not implemented yet!')
            else:
                for i, data in enumerate(zip(val_extra_data['pred'], val_extra_data['true_masks'], val_extra_data['imgs'])):
                    writer.add_images(
                        f'Validation_{validation_step}_ImgBatch[{i}]/img', torch.unsqueeze(data[2], 0), global_step)
                    writer.add_images(
                        f'Validation_{validation_step}_ImgBatch[{i}]/_gt',
                        torch.unsqueeze(torch.unsqueeze(data[1][0, :, :], 0), 0),
                        global_step
                    )
                    writer.add_images(
                        f'Validation_{validation_step}_ImgBatch[{i}]/_pred',
                        torch.unsqueeze(torch.unsqueeze(data[0][0, :, :], 0), 0),
                        global_step
                    )

    def training_step(self, batch):
        """
        Logic to perform the training step per batch

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            pred<torch.Tensor>, true_masks<torch.Tensor>, imgs<torch.Tensor>, loss<List[torch.Tensor]>,
            metrics<List[dict]>, labels<list>, label_names<list>
        """
        assert isinstance(batch, dict), type(batch)
        # TODO: part of these lines can be re-used for get_validation_data with minors tweaks
        #       review if this is a good idea o not
        imgs = batch['image']
        true_masks = batch['mask']
        labels = batch.get('label', ['']*self.train_dataloader_kwargs['batch_size'])
        label_names = batch.get('label_name', ['']*self.train_dataloader_kwargs['batch_size'])

        # commenting out main label validation because at level 1
        # while creating the crops with the desired size some of them
        # could not have data in the main label
        # for i in range(labels.shape[0]):
        #     assert true_masks[i][labels[i]].max() == 1, labels[i].item()

        num_crops = 1

        # TODO: see how to use and process label_names
        imgs, labels, true_masks, _ = self.reshape_data(imgs, labels, true_masks)

        if imgs.shape[1] != self.module.n_channels:
            raise ModelMGRImageChannelsError(self.module.n_channels, imgs.shape[1])

        imgs = imgs.to(device=self.device, dtype=torch.float32)
        # FIXME: review this!!
        # changing this becaue of the error
        # RuntimeError: _thnn_conv_depthwise2d_forward not supported on CUDAType
        #               for Long
        # mask_type = torch.float32 if self.module.n_classes == 1 else torch.long
        # mask_type = torch.float32
        true_masks = true_masks.to(device=self.device, dtype=torch.float32)
        losses = []
        interpolate_mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            masks_pred = self.model(imgs)
            assert isinstance(masks_pred, (list, tuple)), type(masks_pred)

            for idx, masks in enumerate(masks_pred):
                if masks.shape != true_masks.shape:
                    true_masks_ = (F.interpolate(
                        true_masks, size=masks.size()[2:], mode=interpolate_mode, align_corners=False
                    ) >= self.mask_threshold).float()
                else:
                    true_masks_ = true_masks

                # applying lenient masks to first decoder output
                if idx == 0:
                    losses.append(self.calculate_loss(self.criterions, masks, true_masks_*.85))
                else:
                    losses.append(self.calculate_loss(self.criterions, masks, true_masks_))

                # IMPORTANT NOTE:
                # when using online data augmentation, it can return X crops instead of 1, so
                # we need to modify this to loss / (n_val*X)
                # because the loss is result of processing X times more crops than
                # normal, so to properly calculate the final loss we need to divide it by
                # number of batches times X. Here we only divide it by X, the final summation
                # will be divided by num_baches at the training method implementation
                losses[-1] /= num_crops

        metrics = []
        pred = None

        for idx, pred in enumerate(masks_pred, start=1):
            pred = torch.sigmoid(pred) if self.module.n_classes == 1 else torch.softmax(pred, dim=1)
            # FIXME try calculating the metric without the threshold
            pred = (pred > self.mask_threshold).float()
            metrics.append(getattr(self, f'train_metrics{idx}')(pred, true_masks))

        # NOTE: we return the last pred

        return pred, true_masks, imgs, losses, metrics, labels, label_names

    def predict_step(self, patch):
        """
        Returns the prediction of the patch

        Args:
            patch <torch.Tensor>: patch cropped from the input image
        Returns:
            preds_plus_bg <torch.Tensor>
        """
        assert isinstance(patch, torch.Tensor), type(patch)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds = self.model(patch)

        # using last last
        preds = preds[-1]
        preds = preds[0]  # using masks from the only batch returned
        preds = torch.sigmoid(preds) if self.module.n_classes == 1 else torch.softmax(preds, dim=0)

        # adding an extra class full of zeros to represent anything else than the
        # defined classes like background or any other not identified thing
        preds_plus_bg = torch.cat([torch.zeros((1, *preds.shape[1:])), preds.cpu()], dim=0)
        preds_plus_bg[preds_plus_bg <= self.mask_threshold] = 0
        # preds_plus_bg[preds_plus_bg <= 0] = 0
        preds_plus_bg = torch.argmax(preds_plus_bg, dim=0)

        return preds_plus_bg
