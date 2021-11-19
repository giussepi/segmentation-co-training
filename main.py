# -*- coding: utf-8 -*-
""" main """

import logzero

import settings

from utils.patches.patches import ProcessDataset


logzero.loglevel(settings.LOG_LEVEL)


def main():
    ###########################################################################
    #                      Extracting patches from CoNSeP                      #
    ###########################################################################
    db_info = {
        "train": {
            "img": (".png", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Train/Images/"),
            "ann": (".mat", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Train/Labels/"),
        },
        "valid": {
            "img": (".png", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Test/Images/"),
            "ann": (".mat", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Test/Labels/"),
        },
    }

    ProcessDataset(dataset_info=db_info)()


if __name__ == '__main__':
    main()
