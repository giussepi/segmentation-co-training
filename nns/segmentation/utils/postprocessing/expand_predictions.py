# -*- coding: utf-8 -*-
""" nns/segmentation/utils/postprocessing/expand_predictions """

import torch
from logzero import logger
from torch.nn.functional import pad


__all__ = ['ExpandPrediction']


class ExpandPrediction:
    """
    Selects all the annotations/regions from pred that are intersected by pixels from sub_pred
    and returns them as a mask.

    Usage:
        ExpandPrediction()(pred_mask, sub_pred_mask)
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

    def __call__(self, pred, sub_pred):
        return self.process(pred, sub_pred)

    def small_grow(self, pred, sub_pred):
        """
        Grows the sub_mask one pixel in all directions following the pred mask and resturns it

        Kwargs:
            pred     <torch.Tensor>: Predicted mask with all the annotations returned by the model(s)
            sub_pred <torch.Tensor>: Mask containing the disagreement or agreement pixels or any
                                     other of filtered pixels that may or may not have some
                                     intersections with the pred mask

        Returns:
            grown_sub_pred <torch.Tensor>
        """
        assert isinstance(pred, torch.Tensor), type(pred)
        assert isinstance(sub_pred, torch.Tensor), type(sub_pred)

        padded = pad(sub_pred, [1, 1, 1, 1])
        padded *= 2
        height, width = padded.shape
        top_middle = padded[:height-2, 1:width-1]
        middle_left = padded[1:height-1, :width-2]
        middle_middle = sub_pred * 2
        middle_right = padded[1:height-1, 2:]
        bottom_middle = padded[2:, 1:width-1]

        if self.diagonal_pixels:
            top_left = padded[:height-2, :width-2]
            top_right = padded[:height-2, 2:]
            bottom_left = padded[2:, :width-2]
            bottom_right = padded[2:, 2:]

            return ((pred*top_left + pred*top_middle + pred*top_right +
                     pred*middle_left + pred*middle_middle + pred*middle_right +
                     pred*bottom_left + pred*bottom_middle + pred*bottom_right) > 1).float()

        return ((pred*top_middle + pred*middle_left + pred*middle_middle + pred*middle_right
                 + pred*bottom_middle) > 1).float()

    def process(self, pred, sub_pred):
        """
        Selects all the annotations from pred that are intersected by pixels from sub_pred
        and returns them as a mask.

        Kwargs:
            pred     <torch.Tensor>: Predicted mask with all the annotations returned by the model(s)
            sub_pred <torch.Tensor>: Mask containing the disagreement or agreement pixels or any
                                     other of filtered pixels that may or may not have some
                                     intersections with the pred mask

        Returns:
            expanded_subpred <torch.Tensor>
        """
        grown_mask = torch.empty(0)
        new_grown_mask = sub_pred

        iteration = 0

        while not torch.equal(grown_mask, new_grown_mask):
            grown_mask = new_grown_mask
            new_grown_mask = self.small_grow(pred, grown_mask)
            iteration += 1

            if self.debug:
                logger.info(f"Iteration {iteration}:  #####################################")
                logger.info(f"Grown mask: \n {grown_mask}")
                logger.info(f"New grown mask: \n {new_grown_mask}")
                logger.info(f"Pred mask: \n {pred}")

        return grown_mask
