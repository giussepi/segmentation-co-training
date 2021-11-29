# -*- coding: utf-8 -*-
"""
consep/utils/patches/extractor

Source: https://github.com/vqdang/hover_net/blob/master/misc/patch_extractor.py
"""

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from consep.utils.patches.constants import PatchExtractType
from consep.utils.utils import cropping_center


class PatchExtractor:
    """
    Extractor to generate patches with or without padding.
    Turn on debug mode to see how it is done.

    Usage:
        xtractor = PatchExtractor((450, 450), (120, 120))
        img = np.full([1200, 1200, 3], 255, np.uint8)
        patches = xtractor.extract(img)
    """

    def __init__(self, win_size, step_size, patch_type=PatchExtractType.MIRROR, debug=False):
        """
        Initializes the object instance

        Args:
            win_size <tuple>: a tuple of (h, w)
            step_size <tuple>: a tuple of (h, w)
            patch_type <str>: Patch type to extract. See utils.patches.constants.py -> PatchExtractType
                              Default PatchExtractType.MIRROR
            debug <bool>: flag to see how it is done
        """
        self.win_size = win_size
        self.step_size = step_size
        self.patch_type = patch_type
        self.debug = debug
        self.counter = 0

        assert isinstance(self.win_size, tuple), type(self.win_size)
        assert isinstance(self.step_size, tuple), type(self.step_size)
        PatchExtractType.validate(self.patch_type)
        assert isinstance(self.debug, bool), type(self.debug)

    def __get_patch(self, x, ptx):
        pty = (ptx[0] + self.win_size[0], ptx[1] + self.win_size[1])
        win = x[ptx[0]: pty[0], ptx[1]: pty[1]]

        assert (
            win.shape[0] == self.win_size[0] and win.shape[1] == self.win_size[1]
        ), "[BUG] Incorrect Patch Size {0}".format(win.shape)

        if self.debug:
            if self.patch_type == PatchExtractType.MIRROR:
                cen = cropping_center(win, self.step_size)
                cen = cen[..., self.counter % 3]
                cen.fill(150)
            cv2.rectangle(x, ptx, pty, (255, 0, 0), 2)
            plt.imshow(x)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            self.counter += 1
        return win

    def __extract_valid(self, x):
        """Extracted patches without padding, only work in case win_size > step_size.

        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip
        the sliding direction then extract 1 patch starting from right / bottom edge.
        There will be 1 additional patch extracted at the bottom-right corner.

        Args:
            x <np.ndarray>: input image, should be of shape HWC

        Return:
            a list of sub patches, each patch is same dtype as x
        """
        im_h = x.shape[0]
        im_w = x.shape[1]

        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.win_size[1], self.step_size[1])

        sub_patches = []
        # Deal with valid block
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        # Deal with edge case
        if h_flag:
            row = im_h - self.win_size[0]
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if w_flag:
            col = im_w - self.win_size[1]
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if h_flag and w_flag:
            ptx = (im_h - self.win_size[0], im_w - self.win_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)
        return sub_patches

    def __extract_mirror(self, x):
        """Extracted patches with mirror padding the boundary such that the
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image.

        Args:
            x <np.ndarray>: input image, should be of shape HWC
        Return:
            a list of sub patches, each patch is same dtype as x
        """
        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = "constant" if self.debug else "reflect"
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x):
        """
        Extracts the patches based on the patch_type argument

        Args:
            x <np.ndarray>: input image, should be of shape HWC
        """
        if self.patch_type == PatchExtractType.VALID:
            return self.__extract_valid(x)

        if self.patch_type == PatchExtractType.MIRROR:
            return self.__extract_mirror(x)

        raise Exception(f"You have to define how to extract patches for the type {self.patch_type}")
