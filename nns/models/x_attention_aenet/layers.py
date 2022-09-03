# -*- coding: utf-8 -*-
""" nns/models/x_attention_aenet/layers """

from typing import Optional

import torch
from gtorch_utils.nns.models.segmentation.unet.unet_parts import DoubleConv, OutConv,\
    TinyUpAE, TinyAE, MicroUpAE, MicroAE, Down
from torch.nn.modules.batchnorm import _BatchNorm


__all__ = ['AttentionAEConvBlock', 'AttentionAEConvBlock2', 'OutEncoder']


class AttentionAEConvBlock(torch.nn.Module):
    """
    Attention convolutional block for XAttentionAENet using a micro AE embedded in the structure

    Usage:
         class UNet_3Plus_DA(UNet_3Plus):
            def __init__(self, ...):
                super().__init__(in_channels, out_channels)
                # intra-class DA skip-con down3 & gating signal down4 -> up1
                self.up1_with_da = AttentionConvBlock(
                    # attention to skip_connection
                    self.da_block_cls(self.filters[3], self.filters[4] // 2,
                                      # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                                      **self.da_block_config),
                    2*self.filters[3],
                    self.filters[3] // factor,
                    batchnorm_cls=self.batchnorm_cls,
                )
    """

    def __init__(
            self, dablock_obj: torch.nn.Module, conv_in_channels: int, conv_out_channels: int, /, *,
            only_attention: bool = False, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2
    ):
        """
        Kwargs:
            dablock <torch.nn.Module>: Disagreement attention block instance.
                                       e.g ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
            conv_in_channels    <int>: conv_block in channels
            conv_out_channels   <int>: conv_block out channels
            only_attention     <bool>: If true returns only the attention; otherwise, returns the
                                       activation maps with attention. Default False
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            data_dimensions <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                       3 for 3D [batch, channel, depth, height, width]. This argument will
                                       determine to use conv2d or conv3d.
                                       Default 2
        """
        super().__init__()
        self.dattentionblock = dablock_obj
        self.identity = torch.nn.Identity()
        self.only_attention = only_attention
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(dablock_obj, torch.nn.Module), \
            'The provided dablock_obj is not an instance of torch.nn.Module'
        assert isinstance(self.conv_in_channels, int), type(self.conv_in_channels)
        assert isinstance(self.conv_out_channels, int), type(self.conv_out_channels)
        assert isinstance(self.only_attention, bool), type(self.only_attention)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnorm_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        if self.dattentionblock.upsample:
            self.down = Down(self.dattentionblock.m1_act, self.dattentionblock.m1_act,
                             batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions)
            self.up_conv = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
                DoubleConv(self.conv_in_channels, self.conv_out_channels, batchnorm_cls=self.batchnorm_cls,
                           data_dimensions=self.data_dimensions)
            )
            self.maxpool_conv = torch.nn.Sequential(
                maxpoolxd(2),
                DoubleConv(
                    self.conv_out_channels, self.conv_in_channels, batchnorm_cls=self.batchnorm_cls,
                    data_dimensions=self.data_dimensions
                )
            )
        else:
            self.conv_block = DoubleConv(
                self.conv_in_channels, self.conv_out_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /, *, disable_attention: bool = False,
                central_gating: torch.Tensor = None):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
            disable_attention       <bool>: When set to True, identity(x) will be used instead of
                                        dattentionblock(x, skip_connection). Default False
            central_gating  <torch.Tensor>: Gating calculated from the last Down layer (central part of UNet).
                                            Default None
        Returns:
            decoder_x_upsampled<torch.Tensor>, ae_output<Union[torch.Tensor, None]> , ae_input<Union[torch.Tensor, None]>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(disable_attention, bool), type(disable_attention)

        if central_gating is not None:
            assert isinstance(central_gating, torch.Tensor), type(central_gating)

        if disable_attention:
            if central_gating is not None:
                da, att = self.identity(central_gating), None
            else:
                da, att = self.identity(x), None
        else:
            # attention to X
            # da, att = self.dattentionblock(x, skip_connection)
            # attention to skip_connection
            if central_gating is not None:
                da, att = self.dattentionblock(skip_connection, central_gating)
            else:
                da, att = self.dattentionblock(skip_connection, x)

        if self.only_attention:
            return att, None, None

        if self.dattentionblock.upsample:
            # upsampling using the micro AE
            decoder_x = torch.cat((self.down(da), x), dim=1)
            decoder_x_upsampled = self.up_conv(decoder_x)
            decoder_x_updownsampled = self.maxpool_conv(decoder_x_upsampled)

            return decoder_x_upsampled, decoder_x_updownsampled, decoder_x

        decoder_x = torch.cat((da, x), dim=1)
        decoder_x = self.conv_block(decoder_x)

        return decoder_x, None, None


class AttentionAEConvBlock2(torch.nn.Module):
    """
    Attention convolutional block for XAttentionAENet using a Tiny upsampling AE isolated from the NN

    Usage:
         class UNet_3Plus_DA(UNet_3Plus):
            def __init__(self, ...):
                super().__init__(in_channels, out_channels)
                # intra-class DA skip-con down3 & gating signal down4 -> up1
                self.up1_with_da = AttentionConvBlock(
                    # attention to skip_connection
                    self.da_block_cls(self.filters[3], self.filters[4] // 2,
                                      # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                                      **self.da_block_config),
                    2*self.filters[3],
                    self.filters[3] // factor,
                    batchnorm_cls=self.batchnorm_cls,
                )
    """

    def __init__(
            self, dablock_obj: torch.nn.Module, conv_in_channels: int, conv_out_channels: int, /, *,
            only_attention: bool = False, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2
    ):
        """
        Kwargs:
            dablock <torch.nn.Module>: Disagreement attention block instance.
                                       e.g ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
            conv_in_channels    <int>: conv_block in channels
            conv_out_channels   <int>: conv_block out channels
            only_attention     <bool>: If true returns only the attention; otherwise, returns the
                                       activation maps with attention. Default False
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            data_dimensions <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                       3 for 3D [batch, channel, depth, height, width]. This argument will
                                       determine to use conv2d or conv3d.
                                       Default 2
        """
        super().__init__()
        self.dattentionblock = dablock_obj
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.identity = torch.nn.Identity()
        self.only_attention = only_attention
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.dattentionblock, torch.nn.Module), \
            'The provided dablock_obj is not an instance of torch.nn.Module'
        assert isinstance(self.conv_in_channels, int), type(self.conv_in_channels)
        assert isinstance(self.conv_out_channels, int), type(self.conv_out_channels)
        assert isinstance(self.only_attention, bool), type(self.only_attention)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnorm_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        if self.dattentionblock.upsample:
            self.up = torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
            self.up_ae = TinyUpAE(self.dattentionblock.m2_act, self.batchnorm_cls, self.data_dimensions)
            self.conv_block = DoubleConv(
                self.conv_in_channels + self.dattentionblock.m2_act//2, self.conv_out_channels,
                batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions
            )
        else:
            self.conv_block = DoubleConv(
                self.conv_in_channels, self.conv_out_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /, *, disable_attention: bool = False,
                central_gating: torch.Tensor = None):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
            disable_attention       <bool>: When set to True, identity(x) will be used instead of
                                        dattentionblock(x, skip_connection). Default False
            central_gating  <torch.Tensor>: Gating calculated from the last Down layer (central part of UNet).
                                            Default None
        Returns:
            decoder_x<torch.Tensor>, decoded<Union[torch.Tensor, None]>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(disable_attention, bool), type(disable_attention)

        if central_gating is not None:
            assert isinstance(central_gating, torch.Tensor), type(central_gating)

        if disable_attention:
            if central_gating is not None:
                da, att = self.identity(central_gating), None
            else:
                da, att = self.identity(x), None
        else:
            # attention to X
            # da, att = self.dattentionblock(x, skip_connection)
            # attention to skip_connection
            if central_gating is not None:
                da, att = self.dattentionblock(skip_connection, central_gating)
            else:
                da, att = self.dattentionblock(skip_connection, x)

        if self.only_attention:
            return att, None, None

        if self.dattentionblock.upsample:
            # upsampling using the micro AE
            encoded, decoded = self.up_ae(x.detach())
            decoder_x = torch.cat([da, encoded.detach(), self.up(x)], dim=1)
            decoder_x = self.conv_block(decoder_x)

            return decoder_x, decoded

        decoder_x = torch.cat((da, x), dim=1)
        decoder_x = self.conv_block(decoder_x)

        return decoder_x, None


class OutEncoder(torch.nn.Module):
    """
    Autoencoder-based output layer for XAttentionAENet
    """

    SUPPORTED_AE_CLS = (TinyUpAE, TinyAE, MicroUpAE, MicroAE)

    def __init__(self, in_channels: int, out_channels: int, skip_conn_channels: int,
                 batchnorm_cls: Optional[_BatchNorm] = None,
                 data_dimensions: int = 2, ae_cls: torch.nn.Module = MicroUpAE):
        """
        Kwargs:
            in_channels      <int>: in channels (channels of x + channels of skip_connection)
            out_channels     <int>: out channels
            skip_conn_channels <int>: number of channels of the skip connection
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
            ae_cls <torch.nn.Module>: AE class to be used to create the output. It must be one of
                                    the supported Aes (TinyUpAE, TinyAE, MicroUpAE, MicroAE).
                                    Default MicroUpAE
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_conn_channels = skip_conn_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions
        self.ae_cls = ae_cls

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        assert isinstance(self.skip_conn_channels, int), type(self.skip_conn_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert issubclass(self.ae_cls, self.SUPPORTED_AE_CLS), type(self.ae_cls)

        self.refine = DoubleConv(
            self.skip_conn_channels, self.skip_conn_channels, batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions
        )

        if self.ae_cls in (TinyAE, TinyUpAE):
            self.ae = self.ae_cls(self.in_channels, self.batchnorm_cls, self.data_dimensions)
        else:  # MicroUpAE, MicroAE
            self.ae = self.ae_cls(
                self.in_channels, self.in_channels // 2 if self.ae_cls == MicroUpAE else self.in_channels * 2,
                self.batchnorm_cls, self.data_dimensions
            )

        self.out = OutConv(self.in_channels, self.out_channels, self.data_dimensions)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /) -> torch.Tensor:
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps

        Returns:
            output <torch.Tensor>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)

        input_ = torch.cat((self.refine(skip_connection), x), dim=1)
        _, decoded = self.ae(input_)
        output = self.out(decoded)

        return output
