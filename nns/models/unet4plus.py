# -*- coding: utf-8 -*-
""" nns/models/unet4plus.py """

from typing import Optional

import torch
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from torch.nn.modules.batchnorm import _BatchNorm

from nns.models.mixins import InitMixin
from nns.models.unet.utils import unetConvX, UnetUp_CT


__all__ = ['UNet4Plus']


class UNet4Plus(torch.nn.Module, InitMixin):
    """
    Unet4+

    Based on https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_3D.py
    """

    def __init__(
            self, feature_scale: int = 1, n_classes: int = 1, n_channels: int = 1,
            data_dimensions: int = 2, is_batchnorm: bool = True, batchnorm_cls: Optional[_BatchNorm] = None,
            init_type=UNet3InitMethod.KAIMING, dsv: bool = True,
    ):
        """
        Initializes the object instance

        Kwargs:
            feature_scale    <int>: scale factor for the filters. Default 1
            n_channels       <int>: number of channels from the input images. e.g. for RGB use 3. Default 1
            n_classes        <int>: number of classes. Use n_classes=1 for classes <= 2, for the rest or cases
                                    use n_classes = classes. Default 1
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [batch, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width].
                                    Default 2
            is_batchnorm    <bool>: Whether or not use batch normalization. Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            init_type        <int>: Initialization method. Default UNet3InitMethod.KAIMING
            dsv             <bool>: Whether or not apply deep supervision. Default True
        """
        super().__init__()
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.data_dimensions = data_dimensions
        self.is_batchnorm = is_batchnorm
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type
        self.dsv = dsv

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.feature_scale, int), type(self.feature_scale)
        assert self.feature_scale >= 1, 'feature_scale must be bigger or equal to 1'
        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(self.is_batchnorm, bool), type(self.is_batchnorm)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        UNet3InitMethod.validate(self.init_type)
        assert isinstance(self.dsv, bool), type(self.dsv)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        maxpool_kernel_size = (2,) * self.data_dimensions
        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        # encoder #############################################################
        self.conv1 = unetConvX(self.n_channels, filters[0], self.is_batchnorm,
                               data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)
        self.maxpool1 = maxpoolxd(kernel_size=maxpool_kernel_size)
        self.conv2 = unetConvX(filters[0], filters[1], self.is_batchnorm,
                               data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)
        self.maxpool2 = maxpoolxd(kernel_size=maxpool_kernel_size)
        self.conv3 = unetConvX(filters[1], filters[2], self.is_batchnorm,
                               data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)
        self.maxpool3 = maxpoolxd(kernel_size=maxpool_kernel_size)
        self.conv4 = unetConvX(filters[2], filters[3], self.is_batchnorm,
                               data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)
        self.maxpool4 = maxpoolxd(kernel_size=maxpool_kernel_size)
        self.center = unetConvX(filters[3], filters[4], self.is_batchnorm,
                                data_dimensions=self.data_dimensions, batchnorm_cls=self.batchnorm_cls)

        # decoder 1 ###########################################################
        self.up_concat1_1 = UnetUp_CT(filters[1], filters[0], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)

        # decoder 2 ###########################################################
        self.up_concat2_2 = UnetUp_CT(filters[2], filters[1], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)
        self.up_concat2_1 = UnetUp_CT(filters[1], filters[0], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)

        # decoder 3 ###########################################################
        self.up_concat3_3 = UnetUp_CT(filters[3], filters[2], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)
        self.up_concat3_2 = UnetUp_CT(filters[2], filters[1], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)
        self.up_concat3_1 = UnetUp_CT(filters[1], filters[0], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)

        # decoder 4 ###########################################################
        self.up_concat4_4 = UnetUp_CT(filters[4], filters[3], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)
        self.up_concat4_3 = UnetUp_CT(filters[3], filters[2], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)
        self.up_concat4_2 = UnetUp_CT(filters[2], filters[1], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)
        self.up_concat4_1 = UnetUp_CT(filters[1], filters[0], is_batchnorm,
                                      data_dimensions=self.data_dimensions, scale_factor=maxpool_kernel_size)

        if self.dsv:
            # deep supervision ################################################
            self.dsv1 = convxd(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)
            self.dsv2 = convxd(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)
            self.dsv3 = convxd(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)
            self.dsv4 = convxd(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)
            self.outc = convxd(self.n_classes*4,  self.n_classes, kernel_size=1, stride=1, padding=0)
            # TODO: what if I concatenate all the decoder outputs and use a single
            #       large conv to create the final mask
            # self.outc = convxd(filters[0]*4,  self.n_classes, kernel_size=1, stride=1, padding=0)
        else:
            self.outc = convxd(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

        # initializing weights ################################################
        self.initialize_weights(self.init_type, layers_cls=(convxd, self.batchnorm_cls))

    def forward(self, inputs):
        # encoder #############################################################
        en1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(en1)
        en2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(en2)
        en3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(en3)
        en4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(en4)
        center = self.center(maxpool4)

        # decoder 1 ###########################################################
        de1_1 = self.up_concat1_1(en1, en2)  # resolution 1 (original)

        # decoder 2 ###########################################################
        de2_2 = self.up_concat2_2(en2, en3)  # resolution 2
        de2_1 = self.up_concat2_1(de1_1, de2_2)  # resolution 1 (original)

        # decoder 3 ###########################################################
        de3_3 = self.up_concat3_3(en3, en4)  # resolution 3
        de3_2 = self.up_concat3_2(de2_2, de3_3)  # resolution 2
        de3_1 = self.up_concat3_1(de2_1, de3_2)  # resolution 1 (original)

        # decoder 4 ###########################################################
        de4_4 = self.up_concat4_4(en4, center)  # resolution 4
        de4_3 = self.up_concat4_3(de3_3, de4_4)  # resolution 3
        de4_2 = self.up_concat4_2(de3_2, de4_3)  # resolution 2
        de4_1 = self.up_concat4_1(de3_1, de4_2)  # resolution 1 (original)

        if self.dsv:
            # deep supervision ################################################
            dsv_de1_1 = self.dsv1(de1_1)
            dsv_de2_1 = self.dsv2(de2_1)
            dsv_de3_1 = self.dsv3(de3_1)
            dsv_de4_1 = self.dsv4(de4_1)
            # opt 1 : normal DSV
            logits = self.outc(torch.cat([dsv_de1_1, dsv_de2_1, dsv_de3_1, dsv_de4_1], dim=1))
            # opt 2: sum them all and use self.final(mean(summation))
            # opt 3: return 4 dsvs to have 4 losses
            # opt 4: split the NN in 4 isolated modules and handle 4 dsvs separately
        else:
            logits = self.outc(de4_1)

        return logits
