# -*- coding: utf-8 -*-
""" consep/utils/patches/patches """

import glob
import math
import pathlib
import re

import numpy as np
import tqdm
from gutils.decorators import timing

from consep.datasets.constants import Dataset
from consep.datasets.datasets import get_dataset
from consep.utils.patches.constants import PatchExtractType
from consep.utils.patches.extractor import PatchExtractor
from consep.utils.utils import rm_n_mkdir


class ProcessDataset:
    """
    Extracts the patches from a dataset. So far, only CoNSeP is implemented/supported

    Usage:
        db_info = {
            "train": {
                "img": (".png", "<some_path>/dataset/CoNSeP/Train/Images/"),
                "ann": (".mat", "<some_path>/dataset/CoNSeP/Train/Labels/"),
            },
            "valid": {
                "img": (".png", "dataset/CoNSeP/Test/Images/"),
                "ann": (".mat", "dataset/CoNSeP/Test/Labels/"),
            },
        }

        ProcessDataset(dataset_info=db_info)()

    Inspired on: https://github.com/vqdang/hover_net/blob/master/extract_patches.py
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        kwargs:
            win_size <tuple>: patch size (h, w). Default (540, 540)
            step_size <tuple>: step size for the height and width (h, w). Default (164, 164)
            extract_type <str>: Patch type to extract. See utils.patches.constants.py -> PatchExtractType
                                Default PatchExtractType.MIRROR
            type_classification <bool>: Determines whether to extract type map (only applicable to
                                datasets with class labels). Default True
            ann_percentage <float>: percentage of annotations to work with (float number in range ]0, 1]).
                                    Default 1.
            database_name <str>: Name of the dataset to process. See consep.datasets.constants.py -> Dataset
                                 Default Dataset.CoNSeP
            save_root <str>: Path to the directory where the processed dataset will be saved.
                             Default 'dataset/training_data/'
            dataset_info <dict>: Dictionary with the images and annotations details.
                                 Default {
                                             "train": {
                                                 "img": (".png", "dataset/CoNSeP/Train/Images/"),
                                                 "ann": (".mat", "dataset/CoNSeP/Train/Labels/"),
                                             },
                                             "valid": {
                                                 "img": (".png", "dataset/CoNSeP/Test/Images/"),
                                                 "ann": (".mat", "dataset/CoNSeP/Test/Labels/"),
                                         }
            }
        """
        self.win_size = kwargs.get('win_size', (540, 540))
        self.step_size = kwargs.get('step_size', (164, 164))
        self.extract_type = kwargs.get('extract_type', PatchExtractType.MIRROR)
        self.type_classification = kwargs.get('type_classification', True)
        self.ann_percentage = kwargs.get('ann_percentage', 1.)
        self.dataset_name = kwargs.get('dataset_name', Dataset.CoNSeP)
        self.save_root = kwargs.get('save_root', 'dataset/training_data/')
        self.dataset_info = kwargs.get(
            'dataset_info',
            {
                "train": {
                    "img": (".png", "dataset/CoNSeP/Train/Images/"),
                    "ann": (".mat", "dataset/CoNSeP/Train/Labels/"),
                },
                "valid": {
                    "img": (".png", "dataset/CoNSeP/Test/Images/"),
                    "ann": (".mat", "dataset/CoNSeP/Test/Labels/"),
                },
            }
        )

        assert isinstance(self.win_size, tuple), type(self.win_size)
        assert isinstance(self.step_size, tuple), type(self.step_size)
        PatchExtractType.validate(self.extract_type)
        assert isinstance(self.type_classification, bool), type(self.type_classification)
        assert 0 < self.ann_percentage <= 1, 'ann_percentage must be a float number in the range ]0, 1]'
        Dataset.validate(self.dataset_name)
        assert isinstance(self.save_root, str), type(self.save_root)
        assert isinstance(self.dataset_info, dict), type(self.dataset_info)

    def __call__(self):
        self.process()

    @staticmethod
    def patterning(x):
        return re.sub("([\[\]])", "[\\1]", x)

    def select_annotations(self, ann):
        """
        Randomly removes a percentage of annotations based on the self.ann_percentage atttribute

        Args:
            ann <np.ndarray>: NumPy array [height, width, 1] (only instances) or
                              [height, width, 2] (instances and types/classes) containing the
                              annotation values

        Returns:
            selected_annotations <np.ndarray>
        """
        if self.ann_percentage < 1:
            ann_instances = np.unique(ann[..., 0])
            annotations_to_remove = np.random.choice(
                ann_instances,
                math.ceil(len(ann_instances) * (1-self.ann_percentage)),
                replace=False
            )

            # removing chosen annotations instances
            for annotation in annotations_to_remove:
                ann[..., 0][ann[..., 0] == annotation] = 0

            if self.type_classification:
                # removing annotations classes based on the annotations instances
                ann[..., 1][ann[..., 0] == 0] = 0

        return ann

    @timing
    def process(self):
        """ Creates the patches """
        parser = get_dataset(self.dataset_name)
        xtractor = PatchExtractor(self.win_size, self.step_size, self.extract_type)

        for split_name, split_desc in self.dataset_info.items():
            img_ext, img_dir = split_desc["img"]
            ann_ext, ann_dir = split_desc["ann"]

            out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
                self.save_root,
                self.dataset_name,
                split_name,
                self.win_size[0],
                self.win_size[1],
                self.step_size[0],
                self.step_size[1],
            )
            file_list = glob.glob(self.patterning("%s/*%s" % (ann_dir, ann_ext)))
            file_list.sort()  # ensure same ordering across platform
            rm_n_mkdir(out_dir)
            pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbarx = tqdm.tqdm(
                total=len(file_list), bar_format=pbar_format, ascii=True, position=0
            )

            for file_idx, file_path in enumerate(file_list):
                base_name = pathlib.Path(file_path).stem

                img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
                ann = parser.load_ann(
                    "%s/%s%s" % (ann_dir, base_name, ann_ext), self.type_classification
                )
                ann = self.select_annotations(ann)
                img = np.concatenate([img, ann], axis=-1)
                sub_patches = xtractor.extract(img)

                pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
                pbar = tqdm.tqdm(
                    total=len(sub_patches),
                    leave=False,
                    bar_format=pbar_format,
                    ascii=True,
                    position=1,
                )

                for idx, patch in enumerate(sub_patches):
                    np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                    pbar.update()
                pbar.close()

                pbarx.update()
            pbarx.close()
