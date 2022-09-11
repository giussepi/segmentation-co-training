# -*- coding: utf-8 -*-
""" lits17/settings """

from monai import transforms as ts


try:
    import settings as global_settings
except ModuleNotFoundError:
    global_settings = None


DEFAULT_TRANSFORMS = {
    'train': ts.Compose([
        ts.ToTensord(keys=['img', 'mask']),
        # ts.AsChannelFirstd(keys=['img', 'mask'], channel_dim=-1),
        # ts.AddChanneld(keys=['img', 'mask']),
        ts.CropForegroundd(keys=['img', 'mask'], source_key='img', select_fn=lambda x: x > 0),
        ts.RandAxisFlipd(keys=['img', 'mask'], prob=.5),
        ts.RandAffined(
            keys=['img', 'mask'],
            prob=1.,
            rotate_range=0.261799,  # 15 degrees
            translate_range=[0.1*368, 0.1*368, 0.1*368],  # 368 is the image width|height
            scale_range=((-0.3,  0.3), (-0.3, 0.3), (-0.3, 0.3)),
            mode=["bilinear", "nearest"]
        ),
        ts.RandCropByPosNegLabeld(
            keys=['img', 'mask'],
            label_key='mask',
            spatial_size=[96, 160, 160],
            pos=1,
            neg=1,
            num_samples=4,
        ),
        # ts.RandSpatialCropd(
        #     keys=['img', 'mask'], roi_size=[96, 160, 160], random_center=True, random_size=False)
        # ts.AsChannelLastd(keys=['img', 'mask'], channel_dim=1),
        # ts.SqueezeDimd(keys=['img', 'mask'])
    ]),
    'valtest': ts.Compose([
        ts.ToTensord(keys=['img', 'mask']),
        # ts.AsChannelFirstd(keys=['img', 'mask'], channel_dim=-1),
        # ts.AddChanneld(keys=['img', 'mask']),
        ts.CropForegroundd(keys=['img', 'mask'], source_key='img', select_fn=lambda x: x > 0),
        ts.RandCropByPosNegLabeld(
            keys=['img', 'mask'],
            label_key='mask',
            spatial_size=[96, 160, 160],
            pos=1,
            neg=0,
            num_samples=4,
        ),
        # ts.RandSpatialCropd(
        #     keys=['img', 'mask'], roi_size=[96, 160, 160], random_center=False, random_size=False)
        # ts.AsChannelLastd(keys=['img', 'mask'], channel_dim=1),
        # ts.SqueezeDimd(keys=['img', 'mask'])
    ])
}

TRANSFORMS = getattr(global_settings, 'LITS17_TRANSFORMS', DEFAULT_TRANSFORMS)
