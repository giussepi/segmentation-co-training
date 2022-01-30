# -*- coding: utf-8 -*-
""" nns/managers """

from unittest.mock import MagicMock

import torch
from gtorch_utils.nns.managers.exceptions import ModelMGRImageChannelsError
from torch.utils.tensorboard import SummaryWriter

from settings import USE_AMP
from nns.callbacks.plotters.masks import MaskPlotter
from nns.mixins.managers import ModelMGRMixin


class ModelMGR(ModelMGRMixin):
    """  """

    def get_validation_data(self, batch):
        """
        Returns the data to be used for the validation or test

        Args:
            batch <dict>: Dictionary contaning batch data

        Returns:
            imgs<torch.Tensor>, true_masks<torch.Tensor>, masks_pred<tuple of torch.Tensors>, labels<list>, label_names<list>
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

        if len(imgs.shape) == 5:
            # TODO: see how to use and process label_names
            imgs, labels, true_masks, _ = self.reshape_data(imgs, labels, true_masks)

        imgs = imgs.to(device=self.device, dtype=torch.float32)
        # changing this becaue of the error
        # RuntimeError: _thnn_conv_depthwise2d_forward not supported on CUDAType for Long
        # mask_type = torch.float32 if self.module.n_classes == 1 else torch.long
        # mask_type = torch.float32
        true_masks = true_masks.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            masks_pred = self.model(imgs)

        # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # so if we have a tensor then we just put it inside a tuple
        # to not break the workflow
        masks_pred = masks_pred if isinstance(masks_pred, tuple) else (masks_pred, )

        return imgs, true_masks, masks_pred, labels, label_names

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
            apply_threshold <bool>: Whether or not apply thresholding to the predicted mask.
                                Default True

        Returns:
            loss<torch.Tensor>, metric<float>, imgs_counter<int>, extra_data<dict>
        """
        batch = kwargs.get('batch')
        testing = kwargs.get('testing', False)
        plot_to_png = kwargs.get('plot_to_png', False)
        mask_plotter = kwargs.get('mask_plotter', None)
        apply_threshold = kwargs.get('apply_threshold', True)

        assert isinstance(batch, dict), type(batch)
        assert isinstance(testing, bool), type(testing)
        assert isinstance(plot_to_png, bool), type(plot_to_png)
        if mask_plotter:
            assert isinstance(mask_plotter, MaskPlotter), type(mask_plotter)
        assert isinstance(apply_threshold, bool), type(apply_threshold)

        loss = imgs_counter = metric = 0

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            imgs, true_masks, masks_pred, labels, label_names = self.get_validation_data(batch)

            if not testing:
                loss += torch.sum(torch.stack([
                    self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred
                ]))

        # TODO: Try with masks from d5 and other decoders
        pred = masks_pred[0]  # using mask from decoder d1
        pred = torch.sigmoid(pred) if self.module.n_classes == 1 else torch.softmax(pred, dim=1)

        if testing and plot_to_png:
            filenames = tuple(str(imgs_counter + i) for i in range(1, pred.shape[0]+1))
            imgs_counter += pred.shape[0]
            mask_plotter(imgs, true_masks, pred, filenames)

        if apply_threshold:
            # FIXME try calculating the metric without the threshold
            pred = (pred > self.mask_threshold).float()

        metric += self.metric(pred, true_masks).item()

        extra_data = dict(imgs=imgs, pred=pred, true_masks=true_masks, labels=labels, label_names=label_names)

        return loss, metric, imgs_counter, extra_data

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
            pred<torch.Tensor>, true_masks<torch.Tensor>, imgs<torch.Tensor>, loss<torch.Tensor>, metric<float>, labels<list>, label_names<list>
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

        if len(imgs.shape) == 5:
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

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            masks_pred = self.model(imgs)

            # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
            # so if we have a tensor then we just put it inside a tuple
            # to not break the workflow
            masks_pred = masks_pred if isinstance(masks_pred, tuple) else (masks_pred, )

            # NOTE: FOR CROSSENTROPYLOSS I SHOULD BE USING the LOGITS ....
            # summing the losses from the outcome(s) and then backpropagating it
            # __import__("pdb").set_trace()
            # [torch.Size([8, 1, 270, 270]), torch.Size([8, 1, 270, 270]), torch.Size([8, 1, 268, 268]), torch.Size([8, 1, 264, 264]), torch.Size([8, 1, 256, 256])]
            loss = torch.sum(torch.stack([
                self.calculate_loss(self.criterions, masks, true_masks) for masks in masks_pred
            ]))

        # using mask from decoder d1
        # TODO: Try with masks from d5 and other decoders
        pred = masks_pred[0]
        pred = torch.sigmoid(pred) if self.module.n_classes == 1 else torch.softmax(pred, dim=1)

        # FIXME try calculating the metric without the threshold
        pred = (pred > self.mask_threshold).float()
        # AttributeError: 'float' object has no attribute 'item'
        metric = self.metric(pred, true_masks)
        metric = metric if isinstance(metric, float) else self.metric(pred, true_masks).item()

        return pred, true_masks, imgs, loss, metric, labels, label_names

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

        # NOTE: UNet_3Plus_DeepSup returns a tuple of tensor masks
        # so we take the predictions from d1
        # TODO: Try with masks from d5 and other decoders
        preds = preds[0] if isinstance(preds, tuple) else preds
        preds = preds[0]  # using masks from the only batch returned
        preds = torch.sigmoid(preds) if self.module.n_classes == 1 else torch.softmax(preds, dim=0)

        # adding an extra class full of zeros to represent anything else than the
        # defined classes like background or any other not identified thing
        preds_plus_bg = torch.cat([torch.zeros((1, *preds.shape[1:])), preds.cpu()], dim=0)
        preds_plus_bg[preds_plus_bg <= self.mask_threshold] = 0
        # preds_plus_bg[preds_plus_bg <= 0] = 0
        preds_plus_bg = torch.argmax(preds_plus_bg, dim=0)

        return preds_plus_bg
