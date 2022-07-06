# -*- coding: utf-8 -*-
""" nns/models/attention_unet2 """

import torch

from gtorch_utils.nns.models.segmentation.unet.unet_parts import DoubleConv, Down, Up, OutConv

from nns.models.layers.disagreement_attention.basic import AttentionBlock


__all__ = ['AttentionUNet']


class UpConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, /, *, batchnorm_cls=torch.nn.BatchNorm2d):
        super().__init__()

        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DoubleConv(in_channels, out_channels, batchnorm_cls=batchnorm_cls)
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # torch.nn.BatchNorm2d(out_channels),
            # torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionUNet(torch.nn.Module):
    """
    Slighty modified Unet using the encoders and decores without
    the center layers and processing

    Based on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    """

    def __init__(self, n_channels, n_classes, bilinear=True, batchnorm_cls=torch.nn.BatchNorm2d):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batchnorm_cls = batchnorm_cls

        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.bilinear, bool), type(self.bilinear)
        assert issubclass(self.batchnorm_cls, torch.nn.modules.batchnorm._BatchNorm), type(self.batchnom_cls)

        self.inc = DoubleConv(n_channels, 64, batchnorm_cls=self.batchnorm_cls)
        self.down1 = Down(64, 128, self.batchnorm_cls)
        self.down2 = Down(128, 256, self.batchnorm_cls)
        self.down3 = Down(256, 512, self.batchnorm_cls)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, self.batchnorm_cls)
        self.att_conv1 = AttentionBlock(
            512, 1024 // factor, resample=UpConv(1024 // factor, 512, batchnorm_cls=self.batchnorm_cls))
        self.up1 = Up(1024, 512 // factor, bilinear, self.batchnorm_cls)
        self.att_conv2 = AttentionBlock(
            256, 512 // factor, resample=UpConv(512 // factor, 256, batchnorm_cls=self.batchnorm_cls))
        self.up2 = Up(512, 256 // factor, bilinear, self.batchnorm_cls)
        self.att_conv3 = AttentionBlock(
            128, 256 // factor, resample=UpConv(256 // factor, 128, batchnorm_cls=self.batchnorm_cls))
        self.up3 = Up(256, 128 // factor, bilinear, self.batchnorm_cls)
        self.att_conv4 = AttentionBlock(
            64, 128 // factor, resample=UpConv(128 // factor, 64, batchnorm_cls=self.batchnorm_cls))
        self.up4 = Up(128, 64, bilinear, self.batchnorm_cls)
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

        att1, _ = self.att_conv1(x4, x5)
        x = self.up1(x5, att1)
        # print(x.shape)  # [2, 256, 80, 119]
        att2, _ = self.att_conv2(x3, x)
        x = self.up2(x, att2)
        # print(x.shape)  # [2, 128, 160, 239]
        att3, _ = self.att_conv3(x2, x)
        x = self.up3(x, att3)
        # print(x.shape)  # [2, 64, 320, 479]
        att4, _ = self.att_conv4(x1, x)
        x = self.up4(x, att4)
        # print(x.shape)  # [2, 64, 640, 959]
        logits = self.outc(x)
        # print(logits.shape)  # [2, 1, 640, 959]

        return logits
