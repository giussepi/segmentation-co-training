# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/layers """

import torch


__all__ = ['ConvBlock', 'DAConvBlock']


class ConvBlock(torch.nn.Module):
    """ Convolutional block to be used after a disagreement attention block """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class DAConvBlock(torch.nn.Module):
    """
    Disagreement attention convolutional block

    Usage:
         class UNet_3Plus_DA(UNet_3Plus):
            def __init__(self, in_channels=3, out_channels=10):
                super().__init__(in_channels, out_channels)
                # placing disagreement attention between mlpconv1 and mlpconv2
                self.daconvblock1 = DAConvBlock(ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
                # placing disagreement attention between mlpconv2 and mlpconv3
                self.daconvblock2 = DAConvBlock(ThresholdedDisagreementAttentionBlock(192, 192), 384, 192)
    """

    def __init__(self, dablock_obj: torch.nn.Module, conv_in_channels: int, conv_out_channels: int):
        """
        Kwargs:
            dablock <torch.nn.Module>: Disagreement attention block instance.
                                       e.g ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
            conv_in_channels    <int>: ConvBlock in channels
            conv_out_channels   <int>: ConvBlock out channels
        """
        super().__init__()
        assert isinstance(dablock_obj, torch.nn.Module), \
            'The provided dablock_obj is not an instance of torch.nn.Module'
        assert isinstance(conv_in_channels, int), type(conv_in_channels)
        assert isinstance(conv_out_channels, int), type(conv_out_channels)

        self.dattentionblock = dablock_obj
        self.convblock = ConvBlock(conv_in_channels, conv_out_channels)
        self.identity = torch.nn.Identity()

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /, *, disable_attention: bool = False):
        """
        x               <torch.Tensor>: activation/feature maps
        skip_connection <torch.Tensor>: skip connection containing activation/feature maps
        disable_attention       <bool>: When set to True, identity(x) will be used instead of
                                        dattentionblock(x, skip_connection). Default False

        returns torch.Tensor
        """
        if disable_attention:
            da = self.identity(x)
        else:
            da, _ = self.dattentionblock(x, skip_connection)

        x = torch.cat((da, x), dim=1)
        x = self.convblock(x)

        return x
