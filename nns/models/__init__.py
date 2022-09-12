# -*- coding: utf-8 -*-
""" nns/models/__init__ """

from nns.models.attention_unet import AttentionUNet
from nns.models.attention_unet2 import AttentionUNet as AttentionUNet2
from nns.models.deeplab.models.deeplabv3plus import Deeplabv3plus
from nns.models.modular_unet4plus import ModularUNet4Plus
from nns.models.unet.models import UNet2D, UNet_Grid_Attention, UNet_Att_DSV, SingleAttentionBlock, MultiAttentionBlock, UNet3D
from nns.models.unet_3plus_da import UNet_3Plus_DA, UNet_3Plus_DA_Train
from nns.models.unet_3plus_da2 import UNet_3Plus_DA2, UNet_3Plus_DA2_Train
from nns.models.unet_3plus_da2ext import UNet_3Plus_DA2Ext, UNet_3Plus_DA2Ext_Train
from nns.models.unet_3plus_intra_da import UNet_3Plus_Intra_DA
from nns.models.unet_3plus_intra_da_gs import UNet_3Plus_Intra_DA_GS
from nns.models.unet_3plus_intra_da_gs_hdx import UNet_3Plus_Intra_DA_GS_HDX
from nns.models.unet4plus import UNet4Plus
from nns.models.x_attention_aenet import XAttentionAENet
from nns.models.x_attention_unet import XAttentionUNet
from nns.models.x_attention_unet_adsv import XAttentionUNet_ADSV
from nns.models.x_attention_unet_sdsv import XAttentionUNet_SDSV
from nns.models.x_grid_attention_unet import XGridAttentionUNet
