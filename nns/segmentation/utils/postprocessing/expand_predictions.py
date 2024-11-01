# -*- coding: utf-8 -*-
""" nns/segmentation/utils/postprocessing/expand_predictions """

import torch
from logzero import logger
from torch.nn.functional import pad


__all__ = ['ExpandPrediction']


class ExpandPrediction:
    """
    Selects all the annotations/regions from preds that are intersected by pixels from sub_preds
    and returns them as masks.

    Usage:
        ExpandPrediction()(sub_pred_mask, pred_mask)
    """

    def __init__(self, diagonal_pixels=False, debug=False):
        """
        Initializes the object instance

        kwargs:
            diagonal_pixels <bool>: Whether or not consider diagonal pixes when growing the mask
            debug           <bool>: Set to True to print debugging messages
        """
        assert isinstance(diagonal_pixels, bool), type(diagonal_pixels)
        assert isinstance(debug, bool), type(debug)

        self.diagonal_pixels = diagonal_pixels
        self.debug = debug

    def __call__(self, sub_preds, preds):
        return self.process(sub_preds, preds)

    def small_grow(self, sub_preds, preds):
        """
        Grows the sub_mask one pixel in all directions following the pred mask and resturns it

        Kwargs:
            sub_preds <torch.Tensor>: Masks [..., H, W] containing the disagreement or agreement
                                      pixels or any other of filtered pixels that may or may not
                                      have some intersections with the pred masks
            preds     <torch.Tensor>: Predicted masks [..., H, W] with all the annotations returned
                                      by the model(s)

        Returns:
            grown_sub_pred <torch.Tensor> [..., H, W]
        """
        assert isinstance(preds, torch.Tensor), type(preds)
        assert len(preds.shape) >= 2
        assert isinstance(sub_preds, torch.Tensor), type(sub_preds)
        assert len(sub_preds.shape) >= 2

        padded = pad(sub_preds, [1, 1, 1, 1])
        height, width = padded.shape[-2:]
        top_middle = padded[..., :height-2, 1:width-1]
        middle_left = padded[..., 1:height-1, :width-2]
        middle_middle = sub_preds
        middle_right = padded[..., 1:height-1, 2:]
        bottom_middle = padded[..., 2:, 1:width-1]

        if self.diagonal_pixels:
            top_left = padded[..., :height-2, :width-2]
            top_right = padded[..., :height-2, 2:]
            bottom_left = padded[..., 2:, :width-2]
            bottom_right = padded[..., 2:, 2:]

            return (preds*(top_left + top_middle + top_right +
                           middle_left + middle_middle + middle_right +
                           bottom_left + bottom_middle + bottom_right) >= 1).float()

        return (preds*(top_middle + middle_left + middle_middle + middle_right
                       + bottom_middle) >= 1).float()

    def process(self, sub_preds, preds, **kwargs):
        """
        Selects all the annotations from preds that are intersected by pixels from sub_preds
        and returns them as masks.

        Kwargs:
            sub_preds <torch.Tensor>: Masks [..., H, W] containing the disagreement or agreement
                                      pixels or any other of filtered pixels that may or may not
                                      have some intersections with the pred masks. Just to clarify,
                                      sub_preds most of the time should be a subset of preds or at
                                      least it should have some intersection points with preds.
            preds     <torch.Tensor>: Predicted masks [..., H, W] with all the annotations returned
                                      by the model(s)

        Returns:
            expanded_subpred <torch.Tensor>
        """
        grown_mask = torch.empty(0).to(preds.device)
        new_grown_mask = sub_preds

        iteration = 0

        while not torch.equal(grown_mask, new_grown_mask):
            grown_mask = new_grown_mask
            new_grown_mask = self.small_grow(grown_mask, preds)
            iteration += 1

            if self.debug:
                logger.info(f"Iteration {iteration}:  #####################################")
                logger.info(f"Grown mask: \n {grown_mask}")
                logger.info(f"New grown mask: \n {new_grown_mask}")
                logger.info(f"Pred mask: \n {preds}")

        return grown_mask
