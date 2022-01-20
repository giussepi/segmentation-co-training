# -*- coding: utf-8 -*-
""" nns/backbones/settings """

import os

try:
    import settings
except ModuleNotFoundError:
    settings = None

USE_AMP = settings.USE_AMP if hasattr(settings, 'USE_AMP') else False

BN_MOM = settings.BACKBONES_BN_MOM if hasattr(settings, 'BACKBONES_BN_MOM') else .1

MEAN = settings.BACKBONES_MEAN if hasattr(settings, 'BACKBONES_MEAN') else (.485, .456, .406)

STD = settings.BACKBONES_STD if hasattr(settings, 'BACKBONES_STD') else (.229, .224, .225)

DIR_CHECKPOINTS = settings.BACKBONES_DIR_CHECKPOINTS if hasattr(settings, 'BACKBONES_DIR_CHECKPOINTS') else \
    os.path.join('checkpoints', 'backbones')

# NOTE: If using xception, then manually download it from
# https://drive.google.com/file/d/1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi/view
# and place it in the right directory

MODEL_URLS = settings.BACKBONES_MODEL_URLS if hasattr(settings, 'BACKBONES_MODEL_URLS') else {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip',
    'resnet101': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet101s-03a0f310.zip',
    'resnet152': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet152s-36670e8b.zip',
    'xception': os.path.join(DIR_CHECKPOINTS, 'xception_pytorch_imagenet.pth'),
}
