# -*- coding: utf-8 -*-
""" nns/models/modular_unet4plus.py """

from itertools import chain
from typing import Optional, Union, Tuple

import torch
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from torch.nn.modules.batchnorm import _BatchNorm

from nns.models.mixins import InitMixin
from nns.models.unet.utils import unetConvX, UnetUp_CT


__all__ = ['MicroUNet', 'UNetExtension',  'ModularUNet4Plus']


class MicroUNet(torch.nn.Module, InitMixin):
    """
    Micro Unet
    """

    def __init__(
            self, filters: Union[list, tuple] = None, feature_scale: int = 1, n_classes: int = 1,
            n_channels: int = 1, isolate: bool = True, data_dimensions: int = 2, is_batchnorm: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, init_type=UNet3InitMethod.KAIMING
    ):
        """
        Initializes the object instance

        Kwargs:
            filters  <list, tuple>: list of filters for encoder and decoder units. Default = [64, 128]
            feature_scale    <int>: scale factor for the filters. Default 1
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3. Default 1
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes. Default 1
            isolate         <bool>: Whether or returtn detached and cloned tensors in the forward pass.
                                    Default True
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [batch, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            is_batchnorm    <bool>: Whether or not use batch normalization. Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
        """
        super().__init__()
        self.filters = filters if filters is not None else [64, 128]
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.isolate = isolate
        self.data_dimensions = data_dimensions
        self.is_batchnorm = is_batchnorm
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.filters, (list, tuple)), type(self.filters)
        assert len(self.filters) == 2, self.filters
        assert isinstance(self.feature_scale, int), type(self.feature_scale)
        assert self.feature_scale >= 1, 'feature_scale must be bigger or equal to 1'
        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.isolate, bool), type(self.isolate)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(self.is_batchnorm, bool), type(self.is_batchnorm)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)

        self.filters = [int(x / self.feature_scale) for x in self.filters]
        maxpool_kernel_size = (2,) * self.data_dimensions
        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        # encoder #############################################################
        self.conv = unetConvX(self.n_channels, self.filters[0], self.is_batchnorm,
                              data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)
        self.maxpool = maxpoolxd(kernel_size=maxpool_kernel_size)
        self.center = unetConvX(self.filters[0], self.filters[1], self.is_batchnorm,
                                data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)

        # decoder #############################################################
        self.up_concat = UnetUp_CT(self.filters[1], self.filters[0], is_batchnorm,
                                   data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)

        # output ##############################################################
        self.outc = convxd(self.filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

        # initializing weights ################################################
        self.initialize_weights(self.init_type, layers_cls=(convxd, self.batchnorm_cls))

    def forward(self, inputs: torch.Tensor):
        """
        Returns:
            logits <torch.Tensor>, detached_skipt_connections_for_unet_extension <Tuple[torch.Tensors]>
        """
        # encoder #############################################################
        encoder = self.conv(inputs)
        maxpool = self.maxpool(encoder)
        center = self.center(maxpool)

        # decoder #############################################################
        decoder = self.up_concat(encoder, center)

        # output ##############################################################
        logits = self.outc(decoder)

        if self.isolate:
            return logits, (center.detach().clone(), decoder.detach().clone())

        return logits, (center, decoder)


class UNetExtension(torch.nn.Module, InitMixin):
    """
    Create one extra proccesing level for MicroUNet or a previous UNetExtension

    Usage:
        ext1 = UnetExtension([64, 128, 256])
        ext2 = UnetExtension([64, 128, 256, 512])
        ext3 = UnetExtension([64, 128, 256, 512, 1024])
    """

    def __init__(
            self, filters: Union[list, tuple], feature_scale: int = 1, n_classes: int = 1,
            n_channels: int = 1, isolate: bool = True, data_dimensions: int = 2, is_batchnorm: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, init_type=UNet3InitMethod.KAIMING
    ):
        """
        Initializes the object instance

        Kwargs:
            filters  <list, tuple>: complete list of UNet filters to be applied. You must provide at least two.
            feature_scale    <int>: scale factor for the filters. Default 1
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3. Default 1
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes. Default 1
            isolate         <bool>: Whether or returtn detached and cloned tensors in the forward pass.
                                    Default True
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [batch, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            is_batchnorm    <bool>: Whether or not use batch normalization. Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
        """
        super().__init__()
        self.filters = filters
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.isolate = isolate
        self.data_dimensions = data_dimensions
        self.is_batchnorm = is_batchnorm
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.filters, (list, tuple)), type(self.filters)
        assert len(self.filters) >= 2, self.filters
        assert isinstance(self.feature_scale, int), type(self.feature_scale)
        assert self.feature_scale >= 1, 'feature_scale must be bigger or equal to 1'
        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.isolate, bool), type(self.isolate)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(self.is_batchnorm, bool), type(self.is_batchnorm)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)

        self.filters = [int(x / self.feature_scale) for x in self.filters]
        maxpool_kernel_size = (2,) * self.data_dimensions
        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        # encoder #############################################################
        self.maxpool = maxpoolxd(kernel_size=maxpool_kernel_size)
        self.center = unetConvX(filters[-2], filters[-1], self.is_batchnorm,
                                data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)

        # decoder #############################################################
        for i in range(len(self.filters)-1, 0, -1):
            setattr(
                self,
                f'up_concat{i}',
                UnetUp_CT(
                    filters[i], filters[i-1], is_batchnorm,
                    data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size
                )
            )

        # output ##############################################################
        self.outc = convxd(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

        # initializing weights ################################################
        self.initialize_weights(self.init_type, layers_cls=(convxd, self.batchnorm_cls))

    def forward(self, skip_connections: Tuple[torch.Tensor]):
        """
        Kwargs:
            skip_connections <Tuple[torch.Tensor]>: detached and ordered skip connections from
                                                    previous NN module (MicroUnet or UNetExtension)

        Returns:
            logits <torch.Tensor>, detached_skipt_connections_for_unet_extension <Tuple[torch.Tensors]>
        """
        len_skip = len(skip_connections)
        assert len_skip == len(self.filters) - 1

        # encoder #############################################################
        maxpool = self.maxpool(skip_connections[0])
        center = self.center(maxpool)

        # decoder #############################################################
        decoders = [getattr(self, f'up_concat{len_skip}')(skip_connections[0], center)]
        for i, skip in zip(range(len_skip - 1, 0, -1), skip_connections[1:]):
            decoders.append(
                getattr(self, f'up_concat{i}')(skip, decoders[-1])
            )

        logits = self.outc(decoders[-1])

        if self.isolate:
            return logits, tuple(i.detach().clone() for i in chain((center, ), decoders))

        return logits, tuple(i for i in chain((center, ), decoders))


class ModularUNet4Plus(torch.nn.Module, InitMixin):
    """
    Modular Unet4+

    Run it using ModularModelMGR
    """

    def __init__(
            self, filters: Union[list, tuple] = None, feature_scale: int = 1, n_classes: int = 1,
            n_channels: int = 1, isolate: bool = True,
            data_dimensions: int = 2, is_batchnorm: bool = True, batchnorm_cls: Optional[_BatchNorm] = None,
            init_type=UNet3InitMethod.KAIMING
    ):
        """
        Initializes the object instance

        Kwargs:
            filters  <list, tuple>: List of UNet filters. You must provide at least two.
                                    Default [64, 128, 256, 512, 1024]
            feature_scale    <int>: scale factor for the filters. Default 1
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3. Default 1
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes. Default 1
            isolate         <bool>: Whether or not isolate the modules. Default True
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [batch, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            is_batchnorm    <bool>: Whether or not use batch normalization. Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
        """
        super().__init__()
        self.filters = filters if filters is not None else [64, 128, 256, 512, 1024]
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.isolate = isolate
        self.data_dimensions = data_dimensions
        self.is_batchnorm = is_batchnorm
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.filters, (list, tuple)), type(self.filters)
        assert len(self.filters) >= 2, self.filters
        assert isinstance(self.feature_scale, int), type(self.feature_scale)
        assert self.feature_scale >= 1, 'feature_scale must be bigger or equal to 1'
        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.isolate, bool), type(self.isolate)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(self.is_batchnorm, bool), type(self.is_batchnorm)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)

        self.filters = [int(x / self.feature_scale) for x in self.filters]
        self.module_names = ['micro_unet']

        self.micro_unet = MicroUNet(
            self.filters[:2], 1, self.n_classes, self.n_channels, self.isolate, self.data_dimensions,
            self.is_batchnorm, self.batchnorm_cls, self.init_type
        )
        for idx, filter_idx in enumerate(range(3, len(self.filters)+1), start=1):
            setattr(
                self,
                f'ext{idx}',
                UNetExtension(
                    self.filters[:filter_idx], 1, self.n_classes, self.n_channels, self.isolate,
                    self.data_dimensions, self.is_batchnorm, self.batchnorm_cls, self.init_type
                )
            )
            self.module_names.append(f'ext{idx}')

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        logits, skip_connections = self.micro_unet(inputs)
        all_logits = [logits]

        for idx, _ in enumerate(range(3, len(self.filters)+1), start=1):
            logits_, skip_connections = getattr(self, f'ext{idx}')(skip_connections)
            all_logits.append(logits_)

        # opt 1 : normal DSV
        # I dont't think we can apply this because each isolated module must be updated
        # using its own error. Try it anyway to see what could happen
        # logits = self.outc(torch.cat([logits1, logits2, logits3, logits4], dim=1))
        # opt 2: sum them all and use self.final(mean(summation))
        # opt 3: return the summation of all the logits
        # logits = logits1 + logits2 + logits3 + logits4  # this should be equivalent...
        # opt 4: now I think returning a tuple with all the module predictions is the best way
        # keep the modules isolated and working properly

        return tuple(all_logits)
