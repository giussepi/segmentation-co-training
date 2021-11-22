# -*- coding: utf-8 -*-
""" utils/pytorch/utils """

import torch
import torch.nn.functional as F


def apply_padding(inputs0, inputs1):
    """
    Apply padding to match inputs0.size()[2:] with inputs1.size()[:2].
    Inputs format: [batch, channels, height, width]

    Returns padded inputs0

    Args:
        inputs0 (torch.Tensor): tensor 1
        inputs1 (torch.Tensor): tensor 2

    Source:
        https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py#L59
    """
    assert isinstance(inputs0, torch.Tensor)
    assert isinstance(inputs1, torch.Tensor)

    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

    diffY = inputs1.size()[2] - inputs0.size()[2]
    diffX = inputs1.size()[3] - inputs0.size()[3]

    inputs0 = F.pad(
        inputs0,
        [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
    )

    return inputs0
