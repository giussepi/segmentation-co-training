# -*- coding: utf-8 -*-
""" nns/models/unet_3plus_intra_da_gs_hdx
TODO: INCOMPLETE MODULE, THE IMPLEMENTATIONS INSTRUCTIONS ARE IN LINES 352 - 280
"""

from typing import Optional

import torch
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.nns.models.segmentation.unet3_plus.layers import unetConv2
from gtorch_utils.utils.images import apply_padding

from nns.models.layers.disagreement_attention.inter_model import ThresholdedDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.base_disagreement import BaseDisagreementAttentionBlock
from nns.models.layers.disagreement_attention.layers import DAConvBlockGS
from nns.models.mixins import InitMixin


__all__ = ['UNet_3Plus_Intra_DA_GS_HDX']


class UNet_3Plus_Intra_DA_GS_HDX(InitMixin):
    """
    UNet_3Plus with intra-model disagreement attention using gating signal and using the attentions
    during the computation of hdx

    Note: batchnorm_cls and init_type kwargs are copied into da_block_config kwarg
    """

    def __init__(self, da_block_cls: BaseDisagreementAttentionBlock = ThresholdedDisagreementAttentionBlock,
                 da_block_config: Optional[dict] = None, **kwargs):
        """
        Kwargs:
            da_block_cls <BaseDisagreementAttentionBlock>: class descendant of BaseDisagreementAttentionBlock
                                    Default ThresholdedDisagreementAttentionBlock
            da_block_config <dict>: Configuration for disagreement attention block.
                                    Default None
        """
        raise NotImplementedError('this module still need to be implemented!!!')
        super().__init__(**kwargs)
        self._unet3_plust_init(**kwargs)
        assert issubclass(da_block_cls, BaseDisagreementAttentionBlock), \
            f'{da_block_cls} is not a descendant of BaseDisagreementAttentionBlock'
        if da_block_config:
            assert isinstance(da_block_config, dict), type(da_block_config)
            self.da_block_config = da_block_config
        else:
            self.da_block_config = {}
        # adding extra configuration to da_block_config
        self.da_block_config['batchnorm_cls'] = kwargs.get('batchnorm_cls')
        self.da_block_config['init_type'] = kwargs.get('init_type')

        # intra-class DA ga-h4 & skip-con-hd5 -> hd4
        self.intra_da_hd4 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[3], self.filters[4], resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[3] + self.UpChannels,
            self.UpChannels
        )
        # TODO: modify self.conv4d_1 to process the intra_da_hd4, This could improve the results
        # intra-class DA ga-h3 & skip-con-hd4 -> hd3
        self.intra_da_hd3 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[2], self.UpChannels, resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[2] + self.UpChannels,
            self.UpChannels
        )
        # TODO: modify self.self.conv2d_1
        # intra-class DA ga-h2 & skip-conn-hd3 -> hd2
        self.intra_da_hd2 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[1], self.UpChannels, resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[1] + self.UpChannels,
            self.UpChannels
        )
        # TODO: modify self.conv1d_1
        # disagreement attention between h1 and hd1
        # maybe this attention is not necessary... test it
        self.intra_da_hd1 = DAConvBlockGS(
            # attention to skip_connection
            da_block_cls(self.filters[0], self.UpChannels, resample=torch.nn.Upsample(
                scale_factor=2, mode='bilinear'), **self.da_block_config),
            self.filters[0] + self.UpChannels,
            self.UpChannels
        )
        self.initialize_weights(
            kwargs.get('init_type'), layers_cls=(torch.nn.Conv2d, kwargs.get('batchnorm_cls')))

    def _unet3_plust_init(
            self, n_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True,
            batchnorm_cls=torch.nn.BatchNorm2d, init_type=UNet3InitMethod.KAIMING
    ):
        """ creates the NN structure """

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.batchnorm_cls = batchnorm_cls
        self.init_type = init_type
        self.bilinear = not is_deconv

        assert isinstance(self.n_channels, int), type(self.n_channels)
        assert isinstance(self.n_classes, int), type(self.n_classes)
        assert isinstance(self.feature_scale, int), type(self.feature_scale)
        assert isinstance(self.is_deconv, bool), type(self.is_deconv)
        assert isinstance(self.is_batchnorm, bool), type(self.is_batchnorm)
        assert issubclass(self.batchnorm_cls, torch.nn.modules.batchnorm._BatchNorm), type(self.batchnorm_cls)
        UNet3InitMethod.validate(self.init_type)

        self.filters = [64, 128, 256, 512, 1024]

        # -------------Encoder--------------
        self.conv1 = unetConv2(self.n_channels, self.filters[0], self.is_batchnorm,
                               self.batchnorm_cls, init_type=self.init_type)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(self.filters[0], self.filters[1], self.is_batchnorm, self.batchnorm_cls,
                               init_type=self.init_type)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(self.filters[1], self.filters[2], self.is_batchnorm, self.batchnorm_cls,
                               init_type=self.init_type)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(self.filters[2], self.filters[3], self.is_batchnorm, self.batchnorm_cls,
                               init_type=self.init_type)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(self.filters[3], self.filters[4], self.is_batchnorm, self.batchnorm_cls,
                               init_type=self.init_type)

        # -------------Decoder--------------
        self.CatChannels = self.filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        # '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        # array([239.75, 160.  ])
        self.h1_PT_hd4 = torch.nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = torch.nn.Conv2d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = self.batchnorm_cls(self.CatChannels)
        self.h1_PT_hd4_relu = torch.nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        # array([239.75, 160.  ])
        self.h2_PT_hd4 = torch.nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = torch.nn.Conv2d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = self.batchnorm_cls(self.CatChannels)
        self.h2_PT_hd4_relu = torch.nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        # array([239.75, 160.  ])
        self.h3_PT_hd4 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = torch.nn.Conv2d(self.filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = self.batchnorm_cls(self.CatChannels)
        self.h3_PT_hd4_relu = torch.nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        # array([239.75, 160.  ])
        self.h4_Cat_hd4_conv = torch.nn.Conv2d(self.filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = self.batchnorm_cls(self.CatChannels)
        self.h4_Cat_hd4_relu = torch.nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        # array([239.75, 160.  ])
        self.hd5_UT_hd4 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = torch.nn.Conv2d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = self.batchnorm_cls(self.CatChannels)
        self.hd5_UT_hd4_relu = torch.nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = self.batchnorm_cls(self.UpChannels)
        self.relu4d_1 = torch.nn.ReLU(inplace=True)

        # '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        # array([479.5, 320. ])
        self.h1_PT_hd3 = torch.nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = torch.nn.Conv2d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = self.batchnorm_cls(self.CatChannels)
        self.h1_PT_hd3_relu = torch.nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        # array([479.5, 320. ])
        self.h2_PT_hd3 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = torch.nn.Conv2d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = self.batchnorm_cls(self.CatChannels)
        self.h2_PT_hd3_relu = torch.nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        # array([479.5, 320. ])
        self.h3_Cat_hd3_conv = torch.nn.Conv2d(self.filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = self.batchnorm_cls(self.CatChannels)
        self.h3_Cat_hd3_relu = torch.nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        # array([479.5, 320. ])
        self.hd4_UT_hd3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = self.batchnorm_cls(self.CatChannels)
        self.hd4_UT_hd3_relu = torch.nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        # array([479.5, 320. ])
        self.hd5_UT_hd3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = torch.nn.Conv2d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = self.batchnorm_cls(self.CatChannels)
        self.hd5_UT_hd3_relu = torch.nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = self.batchnorm_cls(self.UpChannels)
        self.relu3d_1 = torch.nn.ReLU(inplace=True)

        # '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        # array([959., 640.])
        self.h1_PT_hd2 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = torch.nn.Conv2d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = self.batchnorm_cls(self.CatChannels)
        self.h1_PT_hd2_relu = torch.nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        # array([959., 640.])
        self.h2_Cat_hd2_conv = torch.nn.Conv2d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = self.batchnorm_cls(self.CatChannels)
        self.h2_Cat_hd2_relu = torch.nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        # array([959., 640.])
        self.hd3_UT_hd2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = self.batchnorm_cls(self.CatChannels)
        self.hd3_UT_hd2_relu = torch.nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        # array([959., 640.])
        self.hd4_UT_hd2 = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = self.batchnorm_cls(self.CatChannels)
        self.hd4_UT_hd2_relu = torch.nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        # array([959., 640.])
        self.hd5_UT_hd2 = torch.nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = torch.nn.Conv2d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = self.batchnorm_cls(self.CatChannels)
        self.hd5_UT_hd2_relu = torch.nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = self.batchnorm_cls(self.UpChannels)
        self.relu2d_1 = torch.nn.ReLU(inplace=True)

        # '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        # array([1918, 1280])
        self.h1_Cat_hd1_conv = torch.nn.Conv2d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = self.batchnorm_cls(self.CatChannels)
        self.h1_Cat_hd1_relu = torch.nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        # array([1918., 1280.])
        self.hd2_UT_hd1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = self.batchnorm_cls(self.CatChannels)
        self.hd2_UT_hd1_relu = torch.nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        # array([1918., 1280.])
        self.hd3_UT_hd1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = self.batchnorm_cls(self.CatChannels)
        self.hd3_UT_hd1_relu = torch.nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        # array([1918., 1280.])
        self.hd4_UT_hd1 = torch.nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = self.batchnorm_cls(self.CatChannels)
        self.hd4_UT_hd1_relu = torch.nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        # array([1918., 1280.])
        self.hd5_UT_hd1 = torch.nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = torch.nn.Conv2d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = self.batchnorm_cls(self.CatChannels)
        self.hd5_UT_hd1_relu = torch.nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = self.batchnorm_cls(self.UpChannels)
        self.relu1d_1 = torch.nn.ReLU(inplace=True)

        # output
        self.outconv1 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

    def forward_1(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        h1 = self.conv1(x)  # h1->320*320*64

        return {'h1': h1}

    def forward_2(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h2'] = self.maxpool1(x['h1'])
        x['h2'] = self.conv2(x['h2'])  # h2->160*160*128

        return x

    def forward_3(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h3'] = self.maxpool2(x['h2'])
        x['h3'] = self.conv3(x['h3'])  # h3->80*80*256

        return x

    def forward_4(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h4'] = self.maxpool3(x['h3'])
        x['h4'] = self.conv4(x['h4'])  # h4->40*40*512

        return x

    def forward_5(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h5'] = self.maxpool4(x['h4'])
        x['hd5'] = self.conv5(x['h5'])  # h5->20*20*1024

        return x

    def forward_6(self, x: dict):
        assert isinstance(x, dict), type(x)

        return x

    def forward_7(self, x: dict):
        assert isinstance(x, dict), type(x)

        # TODO: IMPLEMENTATIONS TO BE TRIED (currenlty it's mostly a copy of unet_3plus_intra_da_gs.py)
        # OPTION 1 ############################################################
        #
        # Calulate the attentions
        # e.g. skip-conn h1 & gs h2 -> h1
        #      skip-conn h2 & gs h3 -> h2
        #      skip-conn h3 & gs h4 -> h3
        #      skip-conn h4 & gs hd5 -> h4
        #      hd5_UT_hd4 does not recieve any modification
        # And add them to the concatenation to calculate hd4, of course conv4d_1 needs to
        # be modified. This options does not sound like the best option so could be skipped
        #
        # OPTION 2 ############################################################
        # Add DA to each *_hd4, however that will greatly increase
        # the memory required.... If this is done then the final x['hd4'] won't require any
        # modification
        # e.g. skip-conn h1 & gs h2 -> h1_PT_hd4
        #      skip-conn h2 & gs h3 -> h2_PT_hd4
        #      skip-conn h3 & gs h4 -> h3_PT_hd4
        #      skip-conn h4 & gs hd5 -> h4_Cat_hd4
        #      hd5_UT_hd4 does not recieve any modification
        #
        # option 3 ############################################################
        # Same as options 1 but applying attention to hX
        # e.g. skip-conn h1 & gs h2 -> h1
        #      skip-conn h2 & gs h3 -> h2
        #      skip-conn h3 & gs h4 -> h3
        #      skip-conn h4 & gs hd5 -> h4
        #      hd5_UT_hd4 does not recieve any modification

        x['h1_PT_hd4'] = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(x['h1']))))
        x['h1_PT_hd4'] = apply_padding(x['h1_PT_hd4'], x['h4'])
        x['h2_PT_hd4'] = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(x['h2']))))
        x['h2_PT_hd4'] = apply_padding(x['h2_PT_hd4'], x['h4'])
        x['h3_PT_hd4'] = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(x['h3']))))
        x['h3_PT_hd4'] = apply_padding(x['h3_PT_hd4'], x['h4'])
        x['h4_Cat_hd4'] = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(x['h4'])))
        x['hd5_UT_hd4'] = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(
            self.hd5_UT_hd4(x['hd5']))))
        x['hd5_UT_hd4'] = apply_padding(x['hd5_UT_hd4'], x['h4'])
        x['hd4'] = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((x['h1_PT_hd4'], x['h2_PT_hd4'], x['h3_PT_hd4'], x['h4_Cat_hd4'], x['hd5_UT_hd4']), 1)
        )))  # hd4->40*40*UpChannels
        x['hd4'] = self.intra_da_hd4(x['hd4'], x['h4'], x['hd5'])

        return x

    def forward_8(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_PT_hd3'] = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(x['h1']))))
        x['h1_PT_hd3'] = apply_padding(x['h1_PT_hd3'], x['h3'])
        x['h2_PT_hd3'] = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(x['h2']))))
        x['h2_PT_hd3'] = apply_padding(x['h2_PT_hd3'], x['h3'])
        x['h3_Cat_hd3'] = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(x['h3'])))
        x['hd4_UT_hd3'] = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(
            self.hd4_UT_hd3(x['hd4']))))
        x['hd4_UT_hd3'] = apply_padding(x['hd4_UT_hd3'], x['h3'])
        x['hd5_UT_hd3'] = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(
            self.hd5_UT_hd3(x['hd5']))))
        x['hd5_UT_hd3'] = apply_padding(x['hd5_UT_hd3'], x['h3'])
        x['hd3'] = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((x['h1_PT_hd3'], x['h2_PT_hd3'], x['h3_Cat_hd3'], x['hd4_UT_hd3'], x['hd5_UT_hd3']), 1)
        )))  # hd3->80*80*UpChannels
        x['hd3'] = self.intra_da_hd3(x['hd3'], x['h3'], x['hd4'])

        return x

    def forward_9(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_PT_hd2'] = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(x['h1']))))
        x['h1_PT_hd2'] = apply_padding(x['h1_PT_hd2'], x['h2'])
        x['h2_Cat_hd2'] = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(x['h2'])))
        x['hd3_UT_hd2'] = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(
            self.hd3_UT_hd2(x['hd3']))))
        x['hd3_UT_hd2'] = apply_padding(x['hd3_UT_hd2'], x['h2'])
        x['hd4_UT_hd2'] = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(
            self.hd4_UT_hd2(x['hd4']))))
        x['hd4_UT_hd2'] = apply_padding(x['hd4_UT_hd2'], x['h2'])
        x['hd5_UT_hd2'] = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(
            self.hd5_UT_hd2(x['hd5']))))
        x['hd5_UT_hd2'] = apply_padding(x['hd5_UT_hd2'], x['h2'])
        x['hd2'] = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((x['h1_PT_hd2'], x['h2_Cat_hd2'], x['hd3_UT_hd2'], x['hd4_UT_hd2'], x['hd5_UT_hd2']), 1)
        )))  # hd2->160*160*UpChannels
        x['hd2'] = self.intra_da_hd2(x['hd2'], x['h2'], x['hd3'])

        return x

    def forward_10(self, x: dict):
        assert isinstance(x, dict), type(x)

        x['h1_Cat_hd1'] = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(x['h1'])))
        x['hd2_UT_hd1'] = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(
            self.hd2_UT_hd1(x['hd2']))))
        x['hd2_UT_hd1'] = apply_padding(x['hd2_UT_hd1'], x['h1'])
        x['hd3_UT_hd1'] = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(
            self.hd3_UT_hd1(x['hd3']))))
        x['hd3_UT_hd1'] = apply_padding(x['hd3_UT_hd1'], x['h1'])
        x['hd4_UT_hd1'] = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(
            self.hd4_UT_hd1(x['hd4']))))
        x['hd4_UT_hd1'] = apply_padding(x['hd4_UT_hd1'], x['h1'])
        x['hd5_UT_hd1'] = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(
            self.hd5_UT_hd1(x['hd5']))))
        x['hd5_UT_hd1'] = apply_padding(x['hd5_UT_hd1'], x['h1'])
        x['hd1'] = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((x['h1_Cat_hd1'], x['hd2_UT_hd1'], x['hd3_UT_hd1'], x['hd4_UT_hd1'], x['hd5_UT_hd1']), 1)
        )))  # hd1->320*320*UpChannels
        x['hd1'] = self.intra_da_hd1(x['hd1'], x['h1'], x['hd2'])

        return x

    def forward_11(self, x: dict):
        assert isinstance(x, dict), type(x)

        d1 = self.outconv1(x['hd1'])  # d1->320*320*n_classes

        return d1

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        x = self.forward_1(x)
        x = self.forward_2(x)
        x = self.forward_3(x)
        x = self.forward_4(x)
        x = self.forward_5(x)
        x = self.forward_6(x)
        x = self.forward_7(x)
        x = self.forward_8(x)
        x = self.forward_9(x)
        x = self.forward_10(x)
        x = self.forward_11(x)

        return x
