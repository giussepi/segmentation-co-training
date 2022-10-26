# -*- coding: utf-8 -*-
""" nns/mixins/images_types/ct """

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from gutils.decorators import timing
from gutils.images.images import NIfTI
from gutils.images.processing import get_slices_coords
from logzero import logger
from tqdm import tqdm

from nns.mixins.settings import USE_AMP, DISABLE_PROGRESS_BAR


__all__ = ['CT3DNIfTIMixin']


class CT3DNIfTIMixin:
    """
    Contains methods for ModelMGR that allow creating 3D segmentation labels for
    whole 3D CT NifTI files
    """

    def predict_step(self, patch: torch.Tensor, /, *, pred_idx: int = 0):
        """
        Returns the prediction of the 3D NIfTI CT patch

        Args:
            patch <torch.Tensor>: patch cropped from the input image
            pred_idx       <int>: index of the prediction to be employed. Default 0

        Returns:
            preds <torch.Tensor>
        """
        assert isinstance(patch, torch.Tensor), type(patch)
        assert isinstance(pred_idx, int), type(pred_idx)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds = self.model(patch)

        preds = preds[pred_idx] if isinstance(preds, tuple) else preds
        preds = preds[0]  # using masks from the only batch returned
        preds = torch.sigmoid(preds) if self.module.n_classes == 1 else torch.softmax(preds, dim=0)
        preds = (preds > self.mask_threshold).float()

        return preds

    @timing
    def predict(self, image_path, /, **kwargs):
        """
        Calculates the masks predictions of the 3D NIfTI CT and saves the outcome as a PNG file

        NOTE: This method has only been tested with binary masks

        Kwargs:
            image_path        <str>: path to the image to be analyzed
            preprocess_image <callable>: function to preprocess the image. Default
                                     self.dataset.preprocess
            patch_size   Tuple[int]: patch size [scans, height, width]. Default (32, 80, 80)
            patch_overlapping Tuple[int]: tuple containing overlapping between patches
                                     [scans_overlap, height_overlap, width_overlap].
                                     Its must contain even number;
                                     furthermore, only the 50% of the overlapping is used to create
                                     final mask. This is necessary to remove inconsistent mask borders
                                     between patches. Default (16, 30, 30)
        """
        preprocess_image = kwargs.get('preprocess_image', self.dataset.preprocess)
        patch_size = kwargs.get('patch_size', (32, 80, 80))
        patch_overlapping = kwargs.get('patch_overlapping', (16, 30, 30))

        assert os.path.isfile(image_path), f'{image_path}'
        assert callable(preprocess_image), 'preprocess_image is not a callable'
        assert isinstance(patch_size, tuple), type(patch_size)
        assert len(patch_size) == 3, len(patch_size)
        for i in patch_size:
            assert isinstance(i, int), type(i)
            assert i > 0, i
        assert isinstance(patch_overlapping, tuple), type(patch_overlapping)
        assert len(patch_overlapping) == 3, len(patch_overlapping)
        for i in patch_overlapping:
            assert isinstance(i, int), type(i)
            assert i >= 0, i
            assert i % 2 == 0, 'patch_overlapping must be a even number'

        # Loading the provided checkpoint or the best model obtained during training
        if self.ini_checkpoint:
            _, _ = self.load_checkpoint(self.optimizer(self.model.parameters(), **self.optimizer_kwargs))
        else:
            self.load()

        img = NIfTI(image_path)
        filename = img.meta['name']
        img = img.ndarray  # [height, width, scans]
        y_dim, x_dim, z_dim = img.shape
        final_mask = np.full((y_dim, x_dim, z_dim), 0, dtype='uint8')  # [height, width, scans]
        self.model.eval()
        total_x = len(list(get_slices_coords(x_dim, patch_size[2], patch_overlapping=patch_overlapping[2])))
        total_y = len(list(get_slices_coords(y_dim, patch_size[1], patch_overlapping=patch_overlapping[1])))
        total_z = len(list(get_slices_coords(z_dim, patch_size[0], patch_overlapping=patch_overlapping[0])))
        total = total_x * total_y * total_z
        half_overlapping = [i // 2 for i in patch_overlapping]
        a1 = a2 = b1 = b2 = c1 = c2 = 0

        logger.info('Making predictions')
        with tqdm(total=total, disable=DISABLE_PROGRESS_BAR) as pbar:
            iz_counter = 0
            for iz in get_slices_coords(z_dim, patch_size[0], patch_overlapping=patch_overlapping[0]):
                iy_counter = 0
                for iy in get_slices_coords(y_dim, patch_size[1], patch_overlapping=patch_overlapping[1]):
                    ix_counter = 0
                    for ix in get_slices_coords(x_dim, patch_size[2], patch_overlapping=patch_overlapping[2]):
                        patch = img[iy:iy+patch_size[1], ix:ix+patch_size[2], iz:iz+patch_size[0]]
                        patch = preprocess_image(patch)[0]  # [1, scans, height, width]
                        patch = torch.from_numpy(patch).type(torch.FloatTensor)
                        patch = torch.unsqueeze(patch, 0)  # [1, 1, scans, height, width]
                        patch = patch.to(device=self.device, dtype=torch.float32)
                        # TODO: the pred_idx will change from model to model and
                        #       among Model Managers
                        pred = self.predict_step(patch, pred_idx=0)
                        pred = pred.squeeze().permute(1, 2, 0).detach().cpu()  # [height, width, scans]

                        if iz_counter == 0:
                            a1 = 0
                            a2 = half_overlapping[0]
                        elif iz_counter + 1 == total_z:
                            a1 = half_overlapping[0]
                            a2 = 0
                        else:
                            a1 = a2 = half_overlapping[0]

                        if iy_counter == 0:
                            b1 = 0
                            b2 = half_overlapping[1]
                        elif iy_counter + 1 == total_y:
                            b1 = half_overlapping[1]
                            b2 = 0
                        else:
                            b1 = b2 = half_overlapping[1]

                        if ix_counter == 0:
                            c1 = 0
                            c2 = half_overlapping[2]
                        elif ix_counter + 1 == total_x:
                            c1 = half_overlapping[2]
                            c2 = 0
                        else:
                            c1 = c2 = half_overlapping[2]

                        final_mask[
                            iy+b1:iy+patch_size[1]-b2, ix+c1:ix+patch_size[2]-c2, iz+a1:iz+patch_size[0]-a2
                        ] = pred[b1:patch_size[1]-b2, c1:patch_size[2]-c2, a1:patch_size[0]-a2]
                        ix_counter += 1
                        pbar.update(1)
                    iy_counter += 1
                iz_counter += 1

        NIfTI.save_numpy_as_nifti(final_mask, f'pred_{filename}')

    @staticmethod
    def plot_2D_ct_gt_preds(
            *, ct_path: str, gt_path: str, pred_path: str, only_slices_with_masks: bool = False,
            save_to_disk: bool = False, dpi: int = 300, no_axis: bool = False, tight_layout: bool = True,
            max_slices: int = -1
    ):
        """
        Plot all 2D slices(scans) from the CT, GT and prediction

        Kwargs:
            ct_path   <str>: path to the computed tomography NIfTI file
            gt_path   <str>: path to the ground truth mask NIfTI file
            pred_path <str>: path to the predicted mask NIfTI file
            only_slices_with_masks <bool>: Whether or not plot only slices with predictions.
                             Default False
            save_to_disk <bool>: Whether or not save the plot to disk instead of displaying it
                             Default False
            dpi       <int>: Dots per inch utilised when saving images to disk.
                             Default 300
            no_axis  <bool>: If True the axis is not displayed.
                             Default False
            tight_layout <bool>: Whether or not apply tight_layout over constrained_layout.
                             Default True
            max_slices <int>: Maximum number of slices to plot/save. Set it to -1 to plot them all
                             Default -1
        """
        assert os.path.isfile(ct_path), 'ct_path must point to a valid file'
        assert os.path.isfile(gt_path), 'gt_path must point to a valid file'
        assert os.path.isfile(pred_path), 'pred_path must point to a valid file'
        assert isinstance(only_slices_with_masks, bool), type(only_slices_with_masks)
        assert isinstance(save_to_disk, bool), type(save_to_disk)
        assert isinstance(dpi, int), type(dpi)
        assert isinstance(no_axis, bool), type(no_axis)
        assert isinstance(tight_layout, bool), type(tight_layout)
        assert isinstance(max_slices, int), type(max_slices)
        assert max_slices > 0 or max_slices == -1, max_slices

        ct = NIfTI(ct_path)
        gt = NIfTI(gt_path)
        pred = NIfTI(pred_path)

        assert ct.shape == gt.shape == pred.shape, 'All shapes must be equal'

        def plot_img_mask_pred(scan: int, img: NIfTI, gt: NIfTI, pred: NIfTI):
            fig, axes = plt.subplots(1, 3, constrained_layout=True)

            for axis, image, label in zip(axes.flat, (img, gt, pred), ('CT', 'GT', 'Pred')):
                axis.imshow(image.ndarray[..., scan], cmap='gray')
                axis.set_title(label)
                if no_axis:
                    axis.set_axis_off()

            plt.suptitle(f'slice {scan}')

            if tight_layout:
                plt.tight_layout()

            if save_to_disk:
                plt.savefig(f'scan_{scan}.png', dpi=dpi)
            else:
                plt.show()

            plt.clf()
            plt.close()

        num_slices = ct.shape[-1] if max_slices == -1 else min(ct.shape[-1], max_slices)

        for scan in range(num_slices):
            if only_slices_with_masks and pred.ndarray[..., scan].sum() == 0:
                continue
            plot_img_mask_pred(scan, ct, gt, pred)
