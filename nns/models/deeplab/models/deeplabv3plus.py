# -*- coding: utf-8 -*-
"""
nns/models/deeplab/models/deeplabv3plus

Inspired on https://github.com/YudeWang/semantic-segmentation-codebase/blob/main/lib/net/deeplabv3plus.py
"""

import torch
from torch import nn
import torch.nn.functional as F

from nns.models.deeplab.operators import ASPP
from nns.models.settings import USE_AMP
from nns.backbones import resnet101


__all__ = ['Deeplabv3plus']


class Deeplabv3plus(nn.Module):
    """

    Usage:
        Deeplabv3plus(
            dict(model_aspp_outdim=256,
                 train_bn_mom=3e-4,
                 model_aspp_hasglobal=True,
                 model_shortcut_dim=48,
                 model_num_classes=1,
                 model_freezebn=False,
                 model_channels=3),
            batchnorm=get_batchnorm2d_class(settings.NUM_GPUS), backbone=resnet101,  backbone_pretrained=True,
            dilated=True, multi_grid=False, deep_base=True
        )
    """

    def __init__(self, cfg, backbone=resnet101, batchnorm=nn.BatchNorm2d, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.backbone = backbone
        self.batchnorm = batchnorm

        assert isinstance(self.cfg, dict), type(self.cfg)
        assert isinstance(self.cfg['model_aspp_outdim'], int), type(self.cfg['model_aspp_outdim'])
        assert isinstance(self.cfg['train_bn_mom'], float), type(self.cfg['train_bn_mom'])
        assert isinstance(self.cfg['model_aspp_hasglobal'], bool), type(self.cfg['model_aspp_hasglobal'])
        assert isinstance(self.cfg['model_shortcut_dim'], int), type(self.cfg['model_shortcut_dim'])
        assert isinstance(self.cfg['model_num_classes'], int), type(self.cfg['model_num_classes'])
        assert isinstance(self.cfg['model_freezebn'], bool), type(self.cfg['model_freezebn'])
        assert isinstance(self.cfg['model_channels'], int), type(self.cfg['model_channels'])
        assert isinstance(self.backbone, nn.Module) or callable(self.backbone), \
            'backbone must be an instance of nn.module or a callable'
        assert issubclass(self.batchnorm, object), 'batchnorm must be a class'

        if callable(self.backbone):
            self.backbone = self.backbone(
                pretrained=kwargs.pop('backbone_pretrained', True), norm_layer=self.batchnorm, **kwargs)

        input_channel = self.backbone.OUTPUT_DIM
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=self.cfg['model_aspp_outdim'],
                         rate=[0, 6, 12, 18],
                         bn_mom=self.cfg['train_bn_mom'],
                         has_global=self.cfg['model_aspp_hasglobal'],
                         batchnorm=self.batchnorm)
        # self.dropout1 = nn.Dropout(0.5)

        indim = self.backbone.MIDDLE_DIM
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, self.cfg['model_shortcut_dim'], 3, 1, padding=1, bias=False),
            batchnorm(self.cfg['model_shortcut_dim'], momentum=self.cfg['train_bn_mom'], affine=True),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(self.cfg['model_aspp_outdim']+self.cfg['model_shortcut_dim'], self.cfg['model_aspp_outdim'],
                      3, 1, padding=1, bias=False),
            batchnorm(self.cfg['model_aspp_outdim'], momentum=self.cfg['train_bn_mom'], affine=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv2d(self.cfg['model_aspp_outdim'], self.cfg['model_aspp_outdim'], 3, 1, padding=1, bias=False),
            batchnorm(self.cfg['model_aspp_outdim'], momentum=self.cfg['train_bn_mom'], affine=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(self.cfg['model_aspp_outdim'], self.cfg['model_num_classes'], 1, 1, padding=0)

        for m in self.modules():
            if m not in self.backbone.modules():
                #		if isinstance(m, nn.Conv2d):
                #			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if isinstance(m, batchnorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if self.cfg['model_freezebn']:
            self.freeze_bn()

    @torch.cuda.amp.autocast(enabled=USE_AMP)
    def forward(self, x, getf=False, interpolate=True):
        N, C, H, W = x.size()
        l1, l2, l3, l4 = self.backbone(x)
        feature_aspp = self.aspp(l4)
        # feature_aspp = self.dropout1(feature_aspp)

        feature_shallow = self.shortcut_conv(l1)
        n, c, h, w = feature_shallow.size()
        feature_aspp = F.interpolate(feature_aspp, (h, w), mode='bilinear', align_corners=True)
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature = self.cat_conv(feature_cat)
        result = self.cls_conv(feature)
        result = F.interpolate(result, (H, W), mode='bilinear', align_corners=True)

        if getf:
            if interpolate:
                feature = F.interpolate(feature, (H, W), mode='bilinear', align_corners=True)
            return result, feature

        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, self.batchnorm):
                m.eval()

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, self.batchnorm):
                m.train()

    @property
    def n_classes(self):
        """
        Returns the number of classes (Adaptation to be used with ModelMGR)
        """
        return self.cfg['model_num_classes']

    @property
    def n_channels(self):
        """
        Returns the number of channels. (Adaptation to be used with ModelMGR)

        This works for backbones starting
        """
        return self.cfg['model_channels']
