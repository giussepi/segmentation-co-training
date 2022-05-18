# -*- coding: utf-8 -*-
""" nns/models/__init__ """

from nns.models.attention_unet import AttentionUNet
from nns.models.attention_unet2 import AttentionUNet as AttentionUNet2
from nns.models.deeplab.models.deeplabv3plus import Deeplabv3plus
from nns.models.unet_3plus_da import UNet_3Plus_DA, UNet_3Plus_DA_Train
from nns.models.unet_3plus_da2 import UNet_3Plus_DA2, UNet_3Plus_DA2_Train
from nns.models.unet_3plus_da2ext import UNet_3Plus_DA2Ext, UNet_3Plus_DA2Ext_Train
from nns.models.unet_3plus_intra_da import UNet_3Plus_Intra_DA
