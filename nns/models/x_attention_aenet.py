# -*- coding: utf-8 -*-
""" nns/models/x_attention_aenet """

from collections import namedtuple
from typing import Optional, Tuple, Union, Dict

import torch
from logzero import logger
from gtorch_utils.nns.models.segmentation.unet.unet_parts import DoubleConv, AEDown, AEDown2, OutConv,\
    AEUpConcat, AEUpConcat2, UnetDsv, UnetGridGatingSignal, TinyUpAE, TinyAE, MicroUpAE, MicroAE, \
    Down
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.loss import _Loss

from nns.models.layers.disagreement_attention.intra_model import AttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock

from nns.models.mixins import InitMixin


__all__ = ['AttentionAEConvBlock', 'AttentionAEConvBlock2', 'OutEncoder', 'XAttentionAENet']


Data = namedtuple('Data', ['input', 'output'])


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

    def __init__(self, in_channels: int, out_channels: int, batchnorm_cls: Optional[_BatchNorm] = None,
                 data_dimensions: int = 2, ae_cls: torch.nn.Module = MicroUpAE):
        """
        Kwargs:
            in_channels      <int>: in channels
            out_channels     <int>: out channels
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
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions
        self.ae_cls = ae_cls

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(in_channels, int), type(in_channels)
        assert isinstance(out_channels, int), type(out_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert issubclass(self.ae_cls, self.SUPPORTED_AE_CLS), type(self.ae_cls)

        if self.ae_cls in (TinyAE, TinyUpAE):
            self.ae = self.ae_cls(self.in_channels, self.batchnorm_cls, self.data_dimensions)
        else:  # MicroUpAE, MicroAE
            self.ae = self.ae_cls(
                self.in_channels, self.in_channels // 2 if self.ae_cls == MicroUpAE else self.in_channels * 2,
                self.batchnorm_cls, self.data_dimensions
            )

        self.out = OutConv(self.in_channels, self.out_channels, self.data_dimensions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, decoded = self.ae(x)
        output = self.out(decoded)

        return output


class XAttentionAENet(torch.nn.Module, InitMixin):
    """
    Attention UNet deep-supervised by autoencoders with a replaceable attention unit

    Based on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py and
    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_CT_single_att_dsv_3D.py

    When true_aes = False use MultiPredsModelMGR
         true_aes = True use AEsModelMGR
    """

    def __init__(
            self, n_channels: int, n_classes: int, bilinear: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, init_type=UNet3InitMethod.KAIMING,
            data_dimensions: int = 2,
            da_block_cls: BaseDisagreementAttentionBlock = AttentionBlock,
            da_block_config: Optional[dict] = None,
            dsv: bool = True, isolated_aes: bool = True, true_aes: bool = False,
            aes_loss: Optional[_Loss] = None, out_ae_cls: torch.nn.Module = None
    ):
        """
        Initializes the object instance

        Kwargs:
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes
            bilinear        <bool>: Whether or not use the bilinear mode
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
            da_block_cls <BaseDisagreementAttentionBlock>: Attention block to be used.
                                    Default AttentionBlock (the standard attention for Unet)
            da_block_config <dict>: Configuration for the instance to be created from da_block_cls.
                                    DO NOT PROVIDE 'n_channels' or 'data_dimensions' IN THIS DICTIONARY.
                                    They will be removed because they are properly defined per
                                    AttentionConvBlock. Default None
             dsv            <bool>: Whether or not apply deep supervision. Default True
             isolated_aes   <bool>: Whether or not to use isolated AutoEncoders. Default True,
             true_aes       <bool>: If True the AEs' inputs will be used as targets (normal AEs behaviour),
                                    otherwrise, interpolated masks will be the targets.
                                    Default False
             aes_loss      <_Loss>: Instance of a subclass of _Loss to be used with the autoencoders.
                                    This option is employed only when true_aes is True.
                                    Default torch.nn.MSELoss()
             out_ae_cls <torch.nn.Module>: If provided the output layer will use an AE; otherwise, a
                                    convolution will be used. To see the available options see class
                                    OutEncoder to find out the valid options.
                                    Default None (it means a normal convolution operation will be employed)
        """
        super().__init__()

        # attributes initalization ############################################
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.data_dimensions = data_dimensions
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type
        self.da_block_cls = da_block_cls
        self.dsv = dsv
        self.isolated_aes = isolated_aes
        self.true_aes = true_aes
        self.aes_loss = aes_loss if aes_loss else torch.nn.MSELoss()  # torch.nn.L1Loss()
        self.out_ae_cls = out_ae_cls

        if da_block_config:
            assert isinstance(da_block_config, dict), type(da_block_config)
            self.da_block_config = da_block_config
        else:
            self.da_block_config = {}

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.bilinear, bool), type(self.bilinear)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)
        assert issubclass(self.da_block_cls, BaseDisagreementAttentionBlock), \
            f'{self.da_block_cls} is not a descendant of BaseDisagreementAttentionBlock'
        assert isinstance(self.dsv, bool), type(self.dsv)
        assert isinstance(self.isolated_aes, bool), type(self.isolated_aes)
        assert isinstance(self.true_aes, bool), type(self.true_aes)
        assert isinstance(self.aes_loss, _Loss), type(self.aes_loss)

        # adding extra configuration to da_block_config
        self.da_block_config['batchnorm_cls'] = self.batchnorm_cls
        self.da_block_config['init_type'] = self.init_type

        # removing n_channels from da_block_config, because this value is already
        # defined per AttentionConvBlock
        if 'n_channels' in self.da_block_config:
            logger.warning(
                f'n_channels: {self.da_block_config["n_channels"]} has been removed from '
                'da_block_config'
            )
            self.da_block_config.pop('n_channels')

        if 'data_dimensions' in self.da_block_config:
            logger.warning(
                f'data_dimensions: {self.da_block_config["data_dimensions"]} has been removed from '
                'da_block_config'
            )
            self.da_block_config.pop('data_dimensions')

        self.filters = [64, 128, 256, 512, 1024]

        if self.isolated_aes:
            down = AEDown2
            attention_block = AttentionAEConvBlock2
            up_concat = AEUpConcat2
        else:
            down = AEDown
            attention_block = AttentionAEConvBlock
            up_concat = AEUpConcat

        # Encoder layers ######################################################
        self.inc = DoubleConv(
            n_channels, self.filters[0], batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions
        )
        self.down1 = down(self.filters[0], self.filters[1], self.batchnorm_cls, self.data_dimensions)
        self.down2 = down(self.filters[1], self.filters[2], self.batchnorm_cls, self.data_dimensions)
        self.down3 = down(self.filters[2], self.filters[3], self.batchnorm_cls, self.data_dimensions)
        factor = 2 if self.bilinear else 1
        self.down4 = down(
            self.filters[3], self.filters[4] // factor, self.batchnorm_cls, self.data_dimensions)  # centre
        self.gating = UnetGridGatingSignal(
            self.filters[4] // factor, self.filters[4] // factor, kernel_size=1,
            batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions
        )

        # Decoder layers ######################################################
        # intra-class DA skip-con down3 & gating signal down4 -> up1
        self.up1_with_da = attention_block(
            # attention to skip_connection
            self.da_block_cls(self.filters[3], self.filters[4] // factor,
                              n_channels=self.filters[3],
                              data_dimensions=self.data_dimensions,
                              **self.da_block_config),
            self.filters[3] + (self.filters[4] // factor),
            self.filters[3] // factor,
            batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions,
        )
        # intra-class DA skip conn. down2 & gating signal up1_with_da -> up2
        self.up2_with_da = attention_block(
            # attention to skip_connection
            self.da_block_cls(self.filters[2], self.filters[3] // factor,
                              n_channels=self.filters[2],
                              data_dimensions=self.data_dimensions,
                              **self.da_block_config),
            self.filters[2] + (self.filters[3] // factor),
            self.filters[2] // factor,
            batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions,
        )
        # intra-class DA skip conn. down1 & gating signal up2_with_da -> up3
        self.up3_with_da = attention_block(
            # attention to skip_connection
            self.da_block_cls(self.filters[1], self.filters[2] // factor,
                              n_channels=self.filters[1],
                              data_dimensions=self.data_dimensions,
                              **self.da_block_config),
            self.filters[1] + (self.filters[2] // factor),
            self.filters[1] // factor,
            batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions,
        )
        self.up4 = up_concat(
            self.filters[1] // factor, self.filters[0], self.bilinear, self.batchnorm_cls,
            self.data_dimensions
        )

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        if self.dsv:
            # deep supervision ################################################
            self.dsv1 = UnetDsv(
                in_size=self.filters[3] // factor, out_size=self.n_classes, scale_factor=8,
                data_dimensions=self.data_dimensions
            )
            self.dsv2 = UnetDsv(
                in_size=self.filters[2] // factor, out_size=self.n_classes, scale_factor=4,
                data_dimensions=self.data_dimensions
            )
            self.dsv3 = UnetDsv(
                in_size=self.filters[1] // factor, out_size=self.n_classes, scale_factor=2,
                data_dimensions=self.data_dimensions
            )
            self.dsv4 = convxd(in_channels=self.filters[0], out_channels=self.n_classes, kernel_size=1)
            if self.out_ae_cls:
                self.outc = OutEncoder(self.n_classes*4, self.n_classes, self.batchnorm_cls,
                                       self.data_dimensions, self.out_ae_cls)
            else:
                self.outc = OutConv(self.n_classes*4, self.n_classes, self.data_dimensions)
        else:
            if self.out_ae_cls:
                self.outc = OutEncoder(self.filters[0], self.n_classes, self.batchnom_cls,
                                       self.data_dimensions, self.out_ae_cls)
            else:
                self.outc = OutConv(self.filters[0], self.n_classes, self.data_dimensions)

        if not self.true_aes:
            # encoders outputs ####################################################
            self.ae_down1 = OutConv(self.filters[0], self.n_classes, self.data_dimensions)
            self.ae_down2 = OutConv(self.filters[1], self.n_classes, self.data_dimensions)
            self.ae_down3 = OutConv(self.filters[2], self.n_classes, self.data_dimensions)
            self.ae_down4 = OutConv(self.filters[3], self.n_classes, self.data_dimensions)
            self.ae_up1 = OutConv(
                self.up1_with_da.dattentionblock.m2_act if self.isolated_aes else self.up1_with_da.conv_in_channels,
                self.n_classes, self.data_dimensions
            )
            self.ae_up2 = OutConv(
                self.up2_with_da.dattentionblock.m2_act if self.isolated_aes else self.up2_with_da.conv_in_channels,
                self.n_classes, self.data_dimensions
            )
            self.ae_up3 = OutConv(
                self.up3_with_da.dattentionblock.m2_act if self.isolated_aes else self.up3_with_da.conv_in_channels,
                self.n_classes, self.data_dimensions
            )
            self.ae_up4 = OutConv(
                self.up4.up_ae.in_channels if self.isolated_aes else self.up4.ae.in_channels,
                self.n_classes, self.data_dimensions
            )

            # outputs names [required by MultiPredsModelMGR]
            self.module_names = [
                'ae_down1', 'ae_down2', 'ae_down3', 'ae_down4',
                'ae_up1', 'ae_up2', 'ae_up3', 'ae_up4',
                'outc'
            ]

        # initializing weights ################################################
        self.initialize_weights(self.init_type, layers_cls=(convxd, self.batchnorm_cls))

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, Dict[str, Data]], Tuple[torch.Tensor]]:
        """

        Returns:
            When self.true_aes == True:  Tuple[torch.Tensor, Dict[str, Data]]
            When self.true_aes == False: logits <Tuple[torch.Tensor]>
        """
        # encoder #############################################################
        x1 = self.inc(x)
        x2, x2ae = self.down1(x1)
        x3, x3ae = self.down2(x2)
        x4, x4ae = self.down3(x3)
        x5, x5ae = self.down4(x4)
        gating = self.gating(x5)

        if not self.true_aes:
            logits = [self.ae_down1(x2ae), self.ae_down2(x3ae), self.ae_down3(x4ae), self.ae_down4(x5ae)]

        # decoder ############################################################
        if self.isolated_aes:
            # using AttentionAEConvBlock2
            d1, d1ae = self.up1_with_da(x5, x4, central_gating=gating)
            d2, d2ae = self.up2_with_da(d1, x3)
            d3, d3ae = self.up3_with_da(d2, x2)
            d4, d4ae = self.up4(d3, x1)

            if self.true_aes:
                aes_data = dict(
                    down1=Data(x1, x2ae), down2=Data(x2, x3ae), down3=Data(x3, x4ae), down4=Data(x4, x5ae),
                    up1da=Data(x5, d1ae), up2da=Data(d1, d2ae), up3da=Data(d2, d3ae),
                    up4=Data(d3, d4ae)
                )
            else:
                logits.extend([self.ae_up1(d1ae), self.ae_up2(d2ae), self.ae_up3(d3ae), self.ae_up4(d4ae)])
        else:
            # using AttentionAEConvBlock
            d1, d1ae_out, d1ae_in = self.up1_with_da(x5, x4, central_gating=gating)
            d2, d2ae_out, d2ae_in = self.up2_with_da(d1, x3)
            d3, d3ae_out, d3ae_in = self.up3_with_da(d2, x2)
            d4, d4ae_out, d4ae_in = self.up4(d3, x1)

            if self.true_aes:
                aes_data = dict(
                    down1=Data(x1, x2ae), down2=Data(x2, x3ae), down3=Data(x3, x4ae), down4=Data(x4, x5ae),
                    up1da=Data(d1ae_in, d1ae_out), up2da=Data(d2ae_in, d2ae_out),
                    up3da=Data(d3ae_in, d3ae_out), up4=Data(d4ae_in, d4ae_out)
                )
            else:
                logits.extend([self.ae_up1(d1ae_out), self.ae_up2(d2ae_out),
                               self.ae_up3(d3ae_out), self.ae_up4(d4ae_out)])

        if self.dsv:
            # deep supervision ####################################################
            dsv1 = self.dsv1(d1)
            dsv2 = self.dsv2(d2)
            dsv3 = self.dsv3(d3)
            dsv4 = self.dsv4(d4)

            if self.true_aes:
                logits = self.outc(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
            else:
                logits.append(self.outc(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)))
        else:
            if self.true_aes:
                logits = self.outc(d4)
            else:
                logits.append(self.outc(d4))

        if self.true_aes:
            return logits, aes_data

        return tuple(logits)
