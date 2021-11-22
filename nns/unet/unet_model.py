# -*- coding: utf-8 -*-
""" nns/unet/unet_model """

import torch

from nns.unet.unet_parts import DoubleConv, Down, Up, OutConv


class UNet(torch.nn.Module):
    """
    Slighty modified Unet using the encoders and decores without
    the center layers and processing
    NOTE: Seems to work better than original basic UNet, at least
          in Carvana dataset

    Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)  # [2, 64, 640, 959]
        x2 = self.down1(x1)
        # print(x2.shape)  # [2, 128, 320, 479]
        x3 = self.down2(x2)
        # print(x3.shape)  # [2, 256, 160, 239]
        x4 = self.down3(x3)
        # print(x4.shape)  # [2, 512, 80, 119]
        x5 = self.down4(x4)
        # print(x5.shape)  # [2, 512, 40, 59]
        x = self.up1(x5, x4)
        # print(x.shape)  # [2, 256, 80, 119]
        x = self.up2(x, x3)
        # print(x.shape)  # [2, 128, 160, 239]
        x = self.up3(x, x2)
        # print(x.shape)  # [2, 64, 320, 479]
        x = self.up4(x, x1)
        # print(x.shape)  # [2, 64, 640, 959]
        logits = self.outc(x)
        # print(logits.shape)  # [2, 1, 640, 959]

        # return logits
        return torch.sigmoid(logits)
