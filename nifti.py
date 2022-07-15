# -*- coding: utf-8 -*-

##

import glob
import os
import re
import shutil
import time
from typing import Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom as dicom
import torch
from gutils.decorators import timing
from gutils.files import get_filename_and_extension
from gutils.folders import clean_create_folder
from gutils.numpy_.numpy_ import scale_using_general_min_max_values
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from scipy.ndimage import zoom
from skimage.exposure import equalize_hist, equalize_adapthist
from tqdm import tqdm

##

# CONSTANTS
# set them to 0 to autocalculate the values per image
DICOM_MIN_VAL = -2048
DICOM_MAX_VAL = 3071


class NIfTI:
    """
    Handles basic operations over NIfTi images
    """

    def __init__(self, filepath: str, dtype: np.dtype = np.int16):
        """ Loads and initialzes the object instance """
        self.ndarray = self.meta = None
        self.filepath = filepath
        self.dtype = dtype

        self._load_nifti_img(filepath, dtype)

    def _load_nifti_img(self, filepath: str, dtype: np.dtype = np.int16):
        '''
        NIFTI Image Loader

        Kwargs:
            filepath <tr>: path to the input NIFTI image
            dtype  <type>: dataio type of the nifti numpy array

        Returns:
            img <np.array> meta <dict>
        '''
        assert os.path.isfile(filepath), f'{filepath} does not exist'
        assert isinstance(dtype, type), type(dtype)

        nim = nib.load(filepath)
        self.ndarray = np.array(nim.get_fdata(), dtype=dtype)
        self.ndarray = np.squeeze(self.ndarray)
        self.meta = {'affine': nim.affine,
                     'dim': nim.header['dim'],
                     'pixdim': nim.header['pixdim'],
                     'name': os.path.basename(filepath)
                     }

    def plot_3d_ndarray(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z, x, y = self.ndarray.nonzero()
        ax.scatter(x, y, z, c=z, alpha=1)
        plt.show()

    @property
    def shape(self):
        return self.ndarray.shape

    def clean_3d_ndarray(self, *, height: int = -1, inplace: bool = False):
        """
        Returns only the slices with data
        NOTE: The algorithm assumes that slices with relevant data are contiguous

        Kwargs:
            height   <int>: Desired number of slices with data to be selected.
                            If set to -1, it there will not be a fixed height.
                            Default -1
            inplace <bool>: Whether or not perform the clean operation in the original array. Default False

        Returns:
            cleaned_array <np.ndarray>, selected_data_idx <np.ndarray>
        """
        if height != -1:
            assert isinstance(height, int), type(height)
            assert height <= self.shape[2], \
                'height must be less than or equalt to the total number of NIfTI slices'
        if inplace:
            assert isinstance(inplace, bool), type(inplace)

        selected_data_idx = self.ndarray.sum(axis=0).sum(axis=0).astype(bool)
        # total_cleaned_data_slices = selected_data_idx.sum()
        fist_cleaned_slice_idx = selected_data_idx.argmax()

        if height != -1:
            selected_data_idx = np.zeros_like(selected_data_idx, dtype=bool)

            if fist_cleaned_slice_idx + height >= self.shape[2]:
                slices_with_data = self.ndarray[:, :, -height:]
                selected_data_idx[-height:] = True
            else:
                slices_with_data = self.ndarray[:, :, fist_cleaned_slice_idx:fist_cleaned_slice_idx+height]
                selected_data_idx[fist_cleaned_slice_idx:fist_cleaned_slice_idx+height] = True
        else:
            slices_with_data = self.ndarray[:, :, selected_data_idx]

        if inplace:
            self.ndarray = slices_with_data

        return slices_with_data, selected_data_idx

    def resize(self, target: tuple, *, inplace: bool = False):
        """
        Kwargs:
            target <tuple>: Target dimensions. E.g. (256, 256, 200)
            inplace <bool>: Whether or not perform the resize operation in the original array. Default False

        Returns:
            resized <np.ndarray>
        """
        assert isinstance(target, tuple), type(target)
        assert len(self.shape) == len(target), \
            'The shapes of ndarray and targets must have the same number of elements'

        zoom_factors = np.array(target) / np.array(self.shape)
        resized = zoom(self.ndarray, zoom_factors)
        assert resized.shape == target, 'resize results do not match the target dimensions'

        if inplace:
            self.ndarray = resized

        return resized

    @staticmethod
    def save_numpy_as_nifti(ndarray: np.ndarray, file_path: str = 'new_image.nii.gz'):
        assert isinstance(ndarray, np.ndarray), type(ndarray)
        assert isinstance(file_path, str), type(file_path)

        img = nib.Nifti1Image(ndarray, np.eye(4))
        nib.save(img, file_path)

    def save(self):
        """ Overwrites the file """
        self.save_numpy_as_nifti(self.ndarray, self.filepath)

    def save_as(self, file_path: str = 'new_nifti.nii.gz'):
        self.save_numpy_as_nifti(self.ndarray, file_path)


class DICOM:
    """
    Handles basic operations over DICOM images
    """

    DICOM_EXTENSION = 'dcm'

    def __init__(self, file_path: str):
        """
        Loads the data and initializes the object instance
        """
        assert os.path.isfile(file_path), f'{file_path} does not exists'

        self.file_path = file_path
        self.ndarray = dicom.dcmread(file_path).pixel_array.astype(np.int16)

    def equalize_histogram(self, *, clahe: bool = False, saving_path: str = ''):
        """
        Applies histogram equalization or CLAHE and returns the result as a np.ndarray.

        Kwargs:
            clahe   <bool>: Whether or not apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
                            Default False
            saving_path <str>: full path to save the image. If not provided the equalized image is not
                            saved. Default ''

        Returns:
            equalized_image <np.ndarray>
        """
        assert isinstance(clahe, bool), type(clahe)
        assert isinstance(saving_path, str), type(saving_path)

        rescaled_img = scale_using_general_min_max_values(self.ndarray.astype(
            float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 255))
        img = np.asarray(Image.fromarray(rescaled_img).convert("L"))
        img = equalize_adapthist(img) if clahe else equalize_hist(img)
        img *= 255
        img = np.uint8(img)

        if saving_path:
            Image.fromarray(img).convert("L").save(saving_path)

        return img

    def plot(self):
        plt.imshow(self.ndarray)
        plt.show()

    def resize(self, target: tuple, *, inplace: bool = False):
        """
        Kwargs:
            target <tuple>: Target dimensions. E.g. (256, 256)
            inplace <bool>: Whether or not perform the resize operation in the original array. Default False

        Returns:
            resized <np.ndarray>
        """
        assert isinstance(target, tuple), type(target)
        assert len(self.shape) == len(target), \
            'The shapes of ndarray and targets must have the same number of elements'
        assert isinstance(inplace, bool), type(inplace)

        zoom_factors = np.array(target) / np.array(self.shape)
        resized = zoom(self.ndarray, zoom_factors)
        assert resized.shape == target, 'resize results do not match the target dimensions'

        if inplace:
            self.ndarray = resized

        return resized

    def save_as(self, file_path: str = 'new_image.png', /, *, gray_scale: bool = True):
        """
        Saves the object as a new dcm, jpg, png, tiff, bmp, etc

        Kwargs:
            file_path  <str>: full path to save the image
            gray_scale <boo>: Whether or not save the image in gray scale model.
                              Only applies for non DICOM images.
                              Default True
        """
        assert isinstance(file_path, str), type(file_path)
        assert isinstance(gray_scale, bool), type(gray_scale)

        _, extension = get_filename_and_extension(file_path)

        if extension == self.DICOM_EXTENSION:
            dicom_obj = dicom.dcmread(self.file_path)
            dicom_obj.PixelData = self.ndarray.astype(np.int16).tobytes()
            dicom_obj.Rows, dicom_obj.Columns = self.shape
            dicom_obj.save_as(file_path)
        else:
            rescaled = scale_using_general_min_max_values(self.ndarray.astype(
                float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 255))
            img = Image.fromarray(rescaled)

            if gray_scale and img.mode != 'L':
                img = img.convert("L")
            else:
                img = img.convert('RGB')

            img.save(file_path)

    @property
    def shape(self):
        return self.ndarray.shape


class ProNIfTI(NIfTI):
    """
    Processed NIfTI class
    Creates and provides methods for basic operations over processed NIfTI (pro.nii.gz) files created
    from DICOM files
    """

    EXTENSION = 'pro.nii.gz'

    @staticmethod
    def create_save(dcm_list: Union[list, tuple], /, *, processing: dict = None,
                    saving_path: str = 'new_pronifti.pro.nii.gz'):
        """
        Saves all the DICOMs from the dcm_list as a single ProNIfTI file

        Kwargs:
            dcm_list <list, tuple>: iterable containing the DICOM paths
            processing      <dict>: dictionary with all the DICOM->methods and kwargs to be used.
                                    e.g. dict(resize={'target':(256,256)}). By default all methods receive
                                    inplace=True as keyword argument.
                                    Default None
            saving_path      <str>: full path to save the image. Default new_pronifti.pro.nii.gz

        """
        assert isinstance(dcm_list, (list, tuple)), type(dcm_list)
        assert len(dcm_list) > 0, len(dcm_list)
        if processing:
            assert isinstance(processing, dict), type(processing)
            for method, kwargs in processing.items():
                assert isinstance(method, str), type(method)
                assert isinstance(kwargs, dict), type(kwargs)
        else:
            processing = {}
        assert isinstance(saving_path, str), type(saving_path)

        data = []
        for filepath in dcm_list:
            dicom = DICOM(filepath)
            for method, kwargs in processing.items():
                kwargs['inplace'] = True  # making sure all operations are made in place
                getattr(dicom, method)(**kwargs)
            data.append(dicom.ndarray[:, :, np.newaxis])

        data = np.concatenate(data, axis=2)
        new_nifti = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(new_nifti, saving_path)

    def plot(self, rows: int = 2, cols: int = 2, /):
        """
        Plots a grid of rows * cols images takend from the loaded data

        Kwargs:
            rows <int>: number of rows. Default 2
            cols <int>: number of columns. Default 2
        """
        assert isinstance(rows, int), type(rows)
        assert isinstance(cols, int), type(cols)

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1)

        for axis, img_idx in zip(grid, range(self.shape[-1])):
            axis.imshow(self.ndarray[..., img_idx], cmap='gray')

        plt.show()

    def plot_3d_ndarray(self):
        raise NotImplementedError

    def clean_3d_ndarray(self):
        raise NotImplementedError


class CT82MGR:
    """
    Contains all the logic to methods to process the CT-82 and a dataset that can be consumed by a
    model

    Usage:
        CT82MGR()()
    """

    SAVING_LABELS_FOLDER = 'labels'
    SAVING_CTS_FOLDER = 'images'
    GEN_LABEL_FILENAME_TPL = 'label_{:02d}.nii.gz'
    GEN_CT_FILENAME_TPL = 'CT_{:02}.pro.nii.gz'
    LABEL_REGEX = r'.+label[0]*(?P<id>\d+).nii.gz'
    VERIFICATION_IMG = 'visual_verification.png'

    def __init__(
            self,
            db_path: str = 'CT-82',
            cts_path: str = os.path.join('manifest-1599750808610', 'Pancreas-CT'),
            labels_path: str = 'TCIA_pancreas_labels-02-05-2017',
            saving_path: str = 'CT-82-Pro',
            target_size: tuple = None,
            non_existing_ct_folders: tuple = None
    ):
        """
        Initialized the object instance

        Kwargs:
            db_path     <str>: Path to the main folder containing the dataset. Default 'CT-82'
            cts_path    <str>: path (relative to db_path) to the main folder containing the CT scans.
                               Default 'manifest-1599750808610/Pancreas-CT'
            labels_path <str>: path (relative to db_path) to the main folder containing the labels/masks.
                               Default 'TCIA_pancreas_labels-02-05-2017'
            saving_path <str>: Path to the folder where the processed labels and CTs will be stored
                               Default CT-82-Pro
            target_size <tuple>: target size of the 3D data height x width x slices.
                               Default (368, 368, 96)
            non_existing_ct_folders <tuple>: Tuple containing the id of the non existing CT folders.
                               Default [25, 70].
                               For the Pancreas CT-82 version 2020/09/10 the cases 25 and 70 were
                               found to be from the same scan #2, just cropped slightly differently,
                               so they were removed
                               https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#225140409ddfdf8d3b134d30a5287169935068e3
        """
        self.db_path = db_path
        self.cts_path = os.path.join(self.db_path, cts_path)
        self.labels_path = os.path.join(self.db_path, labels_path)
        self.saving_path = saving_path
        self.target_size = target_size if target_size is not None else (368, 368, 96)
        self.non_existing_ct_folders = \
            non_existing_ct_folders if non_existing_ct_folders is not None else (25, 70)

        assert os.path.isdir(self.db_path), self.db_path
        assert os.path.isdir(self.cts_path), self.cts_path
        assert os.path.isdir(self.labels_path), self.labels_path
        assert isinstance(self.saving_path, str), type(self.saving_path)
        assert isinstance(self.target_size, tuple), type(tuple)
        assert len(self.target_size) == 3, len(self.target_size)
        assert isinstance(self.non_existing_ct_folders, tuple), type(self.non_existing_ct_folders)

        self.saving_labels_folder = os.path.join(self.saving_path, self.SAVING_LABELS_FOLDER)
        self.saving_cts_folder = os.path.join(self.saving_path, self.SAVING_CTS_FOLDER,)
        self.label_pattern = re.compile(self.LABEL_REGEX)

    def __call__(self):
        self.process()

    def _process_labels_cts(self, labels_file: str, cts_folder: str, /):
        """
        Processes the NIfTI labels and DICOM scans provided, then results are saved two NIfTI files

        Kwargs:
            labels_file <str>: path to the NIfTI file to be processed
            cts_folder  <str>: path to the folder containing the DICOM scans corresponding to the
                               provided labels_file
        """
        labels = NIfTI(labels_file)
        _, selected_data_idx = labels.clean_3d_ndarray(height=self.target_size[2], inplace=True)
        # then we apply the resize over the annotated area (ROI)
        labels.resize(self.target_size, inplace=True)
        result = self.label_pattern.match(labels_file)

        if result is None:
            raise RuntimeError(f'The label id from {labels_file} could not be retrieved.')

        label_id = int(result.groupdict()['id'])
        labels.save_as(os.path.join(self.saving_labels_folder, self.GEN_LABEL_FILENAME_TPL.format(label_id)))
        cts_wildcard = os.path.join(cts_folder, '**/*.dcm')
        selected_dicoms = \
            np.array(sorted(glob.glob(cts_wildcard, recursive=True)))[selected_data_idx].tolist()
        ProNIfTI.create_save(
            selected_dicoms,
            processing={'resize': {'target': self.target_size[:2]}},
            saving_path=os.path.join(self.saving_cts_folder, self.GEN_CT_FILENAME_TPL.format(label_id))
        )

    @timing
    def process(self):
        """
        Processes all the labels/masks and CT scans and saves the results
        """
        labels_wildcard = os.path.join(self.labels_path, '*.nii.gz')
        labels_files = glob.glob(labels_wildcard)
        labels_files.sort()
        cts_folders = [os.path.join(self.cts_path, f'PANCREAS_{i:04d}') for i in range(1, 83)]
        cts_folders.sort()

        clean_create_folder(self.saving_labels_folder)
        clean_create_folder(self.saving_cts_folder)

        for idx in self.non_existing_ct_folders[::-1]:
            cts_folders.pop(idx-1)
            labels_files.pop(idx-1)

        for labels_file, cts_folder in zip(labels_files, cts_folders):
            self._process_labels_cts(labels_file, cts_folder)

    def get_min_max_values(self):
        """
        Browse all the dicom files to the minimum and maximum values

        Returns:
            min_, max_, dicoms_analized
        """
        cts_wildcard = os.path.join(self.cts_path, '**/*.dcm')
        dicoms = glob.glob(cts_wildcard, recursive=True)
        dicoms_analized = len(dicoms)
        min_ = np.inf
        max_ = np.NINF

        for dicom_path in dicoms:
            dcm = DICOM(dicom_path)
            min_ = min(min_, dcm.ndarray.min())
            max_ = max(max_, dcm.ndarray.max())

        return min_, max_, dicoms_analized

    def perform_visual_verification(
            self, subject_id: int, /, *, alpha: float = .4, scans: Union[int, list] = 1, clahe: bool = False,
            rgb_mask: tuple = None
    ):
        """
        To see the changes open visual_verification.png and it will be continuosly updated with new mask data

        Kwargs:
            subject_id  <int>: chosen id, without leading zeros, from the generated labels (label_<id>.nii.gz)
                               or CTs (CT_<id>.pro.nii.gz) to be analyzed.
            alpha     <float>: alpha channel value in range ]0, 1]. Default 0.9
            scans <int, list>: If set to an integer it will be the number of scans to be analyzed. If
                               a single-element list is provided it will be the number of scans
                               to be analyzed, if a two-element list is provided it will be the interval
                               of slices to be analyzed. Default 1
            clahe   <bool>: Whether or not apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
                            Default False
            rgb_mask   <bool>: RGB colour for the mask. Default (148, 36, 36)
        """
        rgb_mask = rgb_mask if rgb_mask else (148, 36, 36)

        assert isinstance(subject_id, int), type(subject_id)
        assert 0 < alpha <= 1, alpha
        assert isinstance(scans, (int, list)), type(scans)
        if isinstance(scans, list):
            assert len(scans) <= 2, 'scan list can have two elements at most'
            if len(scans) == 2:
                assert scans[0] < scans[1]
        assert isinstance(clahe, bool), type(clahe)
        assert isinstance(rgb_mask, tuple), type(rgb_mask)

        label = NIfTI(os.path.join(self.saving_labels_folder, self.GEN_LABEL_FILENAME_TPL.format(subject_id)))
        ct = ProNIfTI(os.path.join(self.saving_cts_folder, self.GEN_CT_FILENAME_TPL.format(subject_id)))

        scans = [scans] if isinstance(scans, int) else [*range(*scans)]

        for scan in scans:
            mask = label.ndarray[..., scan] * 255
            image = scale_using_general_min_max_values(ct.ndarray[..., scan].astype(
                float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 255))

            if clahe:
                image = np.asarray(Image.fromarray(np.uint8(image)).convert('L'))
                image = equalize_adapthist(image)*255
                image = image.astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image.astype(np.uint8)).convert('RGB')

            mask = Image.fromarray(mask.T.astype(np.uint8)).convert('RGBA')
            new_data = []

            # setting black as transparent
            for item in mask.getdata():
                if item[0] == item[1] == item[2] == 0:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append((*rgb_mask, 255))

            mask.putdata(new_data)
            mask_trans = Image.new("RGBA", mask.size)
            mask_trans = Image.blend(mask_trans, mask, alpha)
            image.save(self.VERIFICATION_IMG)
            time.sleep(1)
            image.paste(mask_trans, (0, 0), mask_trans)
            image.save(self.VERIFICATION_IMG)
            time.sleep(1)
            # plt.imshow(np.asarray(image))
            # plt.pause(1)
            # plt.close()


def test_NIfTI_clean_3d_ndarray():
    fist_cleaned_slice_idx = 99
    slices_with_data = 71
    path = 'label0001.nii.gz'
    n = NIfTI(path, np.int16)
    original_num_slices = n.shape[2]

    arr, idx = n.clean_3d_ndarray(inplace=False)
    assert arr.shape[2] == slices_with_data
    assert arr.shape != n.shape
    assert arr.sum(0).sum(0).astype(bool).sum() == slices_with_data
    assert idx.shape[0] == original_num_slices
    assert idx[fist_cleaned_slice_idx:fist_cleaned_slice_idx+slices_with_data].sum() == slices_with_data
    assert idx.sum() == slices_with_data

    arr, idx = n.clean_3d_ndarray(height=55, inplace=False)
    assert arr.shape[2] == 55
    assert arr.shape != n.shape
    assert arr.sum(0).sum(0).astype(bool).sum() == 55
    assert idx.shape[0] == original_num_slices
    assert idx[fist_cleaned_slice_idx:fist_cleaned_slice_idx+55].sum() == 55
    assert idx.sum() == 55

    arr, idx = n.clean_3d_ndarray(height=100, inplace=False)
    assert arr.shape[2] == 100
    assert arr.shape != n.shape
    assert arr.sum(0).sum(0).astype(bool).sum() == slices_with_data
    assert idx.shape[0] == original_num_slices
    assert idx[fist_cleaned_slice_idx:fist_cleaned_slice_idx+100].sum() == 100
    assert idx.sum() == 100

    arr, idx = n.clean_3d_ndarray(height=230, inplace=False)
    assert arr.shape[2] == 230
    assert arr.shape != n.shape
    assert arr.sum(0).sum(0).astype(bool).sum() == slices_with_data
    assert idx.shape[0] == original_num_slices
    assert idx[-230:].sum() == 230
    assert idx.sum() == 230

    try:
        arr, idx = n.clean_3d_ndarray(height=241, inplace=False)
    except AssertionError:
        pass

    arr, idx = n.clean_3d_ndarray(inplace=True)
    assert n.shape[2] == slices_with_data
    assert id(arr) == id(n.ndarray)
    assert n.ndarray.sum(0).sum(0).astype(bool).sum() == slices_with_data
    assert idx.shape[0] == original_num_slices
    assert idx.shape[0] != n.shape[2]
    assert idx[fist_cleaned_slice_idx:fist_cleaned_slice_idx+slices_with_data].sum() == slices_with_data
    assert idx.sum() == slices_with_data


def test_DICOM_equalize_historgram():
    img = DICOM('1-001.dcm')
    equalized = img.equalize_histogram()
    assert not np.array_equal(equalized, img.ndarray)
    assert equalized.shape == img.shape
    assert 0 <= equalized.min() <= 255
    assert 0 <= equalized.max() <= 255
    assert not os.path.isfile('1-001.png')

    equalized_clahe = img.equalize_histogram(clahe=True, saving_path='1-001.png')
    assert not np.array_equal(equalized, equalized_clahe)
    assert not np.array_equal(equalized_clahe, img.ndarray)
    assert equalized_clahe.shape == img.shape
    assert 0 <= equalized_clahe.min() <= 255
    assert 0 <= equalized_clahe.max() <= 255
    assert os.path.isfile('1-001.png')
    os.remove('1-001.png')


def test_ProNIfTI_create_save():
    files = ['1-001.dcm', '1-002.dcm']
    ProNIfTI.create_save(files)
    data = ProNIfTI('new_pronifti.pro.nii.gz')
    assert data.shape == (512, 512, 2), data.shape
    os.remove('new_pronifti.pro.nii.gz')

    ProNIfTI.create_save(files, saving_path='xuxuca.pro.nii.gz')
    data = ProNIfTI('xuxuca.pro.nii.gz')
    assert data.shape == (512, 512, 2), data.shape
    os.remove('xuxuca.pro.nii.gz')

    ProNIfTI.create_save(files, processing={'resize': {'target': (256, 256)}})
    data = ProNIfTI('new_pronifti.pro.nii.gz')
    assert data.shape == (256, 256, 2), data.shape
    os.remove('new_pronifti.pro.nii.gz')


def test_ProNIfTi_plot():
    files = [f'1-{i:03d}.dcm' for i in range(1, 6)]
    ProNIfTI.create_save(files, processing={'resize': {'target': (256, 256)}})
    data = ProNIfTI('new_pronifti.pro.nii.gz')
    data.plot()
    data.plot(1, 1)
    data.plot(2, 3)
    os.remove('new_pronifti.pro.nii.gz')


def test_CT82MGR_process():
    target_size = (1024, 1024, 96)  # (160, 160, 96)
    mgr = CT82MGR(
        db_path=os.path.join('test_datasets', 'CT-82'),
        cts_path='images',
        labels_path='labels',
        saving_path='test_CT-82-pro',
        target_size=target_size
    )
    mgr.non_existing_ct_folders = []
    mgr()
    assert len(glob.glob(os.path.join(mgr.saving_labels_folder, r'*.nii.gz'))) == 2
    assert len(glob.glob(os.path.join(mgr.saving_cts_folder, r'*.pro.nii.gz'))) == 2

    for subject in range(1, 3):
        labels = NIfTI(os.path.join(mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
        cts = ProNIfTI(os.path.join(mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
        assert labels.shape == cts.shape == target_size

    shutil.rmtree(mgr.saving_path)


def test_CT82MGR_perform_visual_verification():
    target_size = (160, 160, 96)  # (1024, 1024, 96)
    mgr = CT82MGR(
        db_path=os.path.join('test_datasets', 'CT-82'),
        cts_path='images',
        labels_path='labels',
        saving_path='test_CT-82-pro',
        target_size=target_size
    )
    mgr.non_existing_ct_folders = []
    mgr()
    mgr.perform_visual_verification(1, scans=1, clahe=True)
    # mgr.perform_visual_verification(1, scans=[72], clahe=True)
    assert os.path.isfile(mgr.VERIFICATION_IMG)
    os.remove(mgr.VERIFICATION_IMG)
    shutil.rmtree(mgr.saving_path)


def test_CT82MGR_get_min_max_values():
    target_size = (1024, 1024, 96)  # (160, 160, 96)
    mgr = CT82MGR(
        db_path=os.path.join('test_datasets', 'CT-82'),
        cts_path='images',
        labels_path='labels',
        saving_path='test_CT-82-pro',
        target_size=target_size
    )
    mgr.non_existing_ct_folders = []
    min_, max_, dicoms_analized = mgr.get_min_max_values()

    assert dicoms_analized == 872, dicoms_analized
    assert min_ == -1024, min_
    assert max_ == 2421, max_


# test_NIfTI_clean_3d_ndarray()
# test_DICOM_equalize_historgram()
# test_ProNIfTI_create_save()
# test_ProNIfTi_plot()
# test_CT82MGR()
# test_CT82MGR_perform_visual_verification()
# test_CT82MGR_get_min_max_values()

##
path = 'label0001.nii.gz'
n = NIfTI(path, np.int16)
print(n.ndarray.shape)
print(n.ndarray.min(), n.ndarray.max())
n.plot_3d_ndarray()
##
print(f'raw {n.ndarray.shape}')
n.clean_3d_ndarray(height=10, inplace=True)
print(f'cleaned {n.ndarray.shape}')
n.resize((368, 368, n.ndarray.shape[2]), inplace=True)
print(f'cleaned & resized {n.ndarray.shape}')
print(n.ndarray.min(), n.ndarray.max())
# plot_3d_ndarray(cleaned_n)
n.save_as('new2.nii.gz')
ct = NIfTI('new2.nii.gz')
ct.plot_3d_ndarray()

##

image_path = '1-001.dcm'
# ds = dicom.dcmread(image_path)
# plt.imshow(ds.pixel_array)
# plt.show()
ds = DICOM(image_path)
print(ds.ndarray.min(), ds.ndarray.max())
# ds.plot()
print(ds.shape)
ds.resize((368, 368), inplace=True)
print(ds.shape)
# ds.plot()
ds.save_as('new_image.dcm', gray_scale=True)
ds.save_as('new_image.png', gray_scale=True)

##
# testing DICOM->equalize_histogram
image = DICOM('1-001.dcm')
# image.equalize_histogram(clahe=True, times_255=True, inplace=True)
eq_img = image.equalize_histogram(clahe=True, saving_path="new_image_equalized_adapthist.png")
plt.imshow(eq_img, cmap='gray')
plt.show()

##
# saving numpy array as NIfTI
# data = np.arange(4*4*3, dtype=np.int16).reshape(4, 4, 3)
data1 = np.expand_dims(DICOM('1-001.dcm').ndarray, axis=2)
data2 = np.expand_dims(DICOM('1-002.dcm').ndarray, axis=2)
data = np.concatenate([data1, data2], axis=2)
new_image = nib.Nifti1Image(data, affine=np.eye(4))
new_image_data = np.squeeze(np.array(new_image.get_fdata(), np.int16))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z, x, y = new_image_data.nonzero()
ax.scatter(x, y, z, c=z, alpha=1)
plt.show()


nib.save(new_image, 'xuxuca.nii.gz')
ii = NIfTI('xuxuca.nii.gz')
ii.plot_3d_ndarray()
ii.ndarray.min(), ii.ndarray.max()
plt.imshow(ii.ndarray[:, :, 0], cmap='gray')
plt.show()
##


image_path2 = '/home/giussepi/Downloads/CT-82/manifest-1599750808610/Pancreas-CT/PANCREAS_0002/11-24-2015-PANCREAS0002-Pancreas-23046/Pancreas-63502/1-001.dcm'
ds2 = dicom.dcmread(image_path2)
##

# scaler = MinMaxScaler(clip=True)
matrix = np.stack([ds.pixel_array, ds2.pixel_array], axis=0)
view = matrix.reshape(2, -1)
# scaler.fit(view)
print(view.min(), view.max())
# rescaled = scaler.transform(view)
rescaled = scale_using_general_min_max_values(view.astype(
    float), min_val=DICOM_MIN_VAL, max_val=DICOM_MAX_VAL, feats_range=(0, 1))
print(rescaled.min(), rescaled.max())
print(rescaled.reshape(2, *ds.pixel_array.shape))

# saving StandardScaler
# https://stackoverflow.com/questions/53152627/saving-standardscaler-model-for-use-on-new-datasets

##
# TOTAL NUMBER OF DICOM AND NIFTI FILES #######################################
dataset_path = 'CT-82'
images_regex = os.path.join(dataset_path, '**/*.dcm')
# images_total = len(glob.glob(images_regex, recursive=True))
# print(f"total number of DICOM files: {images_total}")  # 18942

labels_path = os.path.join(dataset_path, 'TCIA_pancreas_labels-02-05-2017')
labels_regex = os.path.join(labels_path, '*.nii.gz')
# labels_total = len(glob.glob(labels_regex))
# print(f"total number of NIfTI files: {labels_total}")  # 82

# DICOMS PER SUBJECT ##########################################################
path_to_subject_DICOMs = os.path.join(dataset_path, 'manifest-1599750808610', 'Pancreas-CT')
dicoms_per_subject = []
aaa = []
for i in range(1, 83):
    subject_folder = os.path.join(path_to_subject_DICOMs, f'PANCREAS_{i:04d}')
    subject_regex = os.path.join(subject_folder, '**/*.dcm')
    dicoms_per_subject.append(len(glob.glob(subject_regex, recursive=True)))
    aaa.append((dicoms_per_subject[-1], subject_regex))
# removing empty folders
dicoms_per_subject.pop(69)
dicoms_per_subject.pop(24)
print(f'DICOMS PER SUBJECT: MIN {min(dicoms_per_subject)}, MAX {max(dicoms_per_subject)}')

# MIN MAX DICOMS WITH LABELS ##################################################

total_dicoms_per_subject = []
cleaned_dicoms_per_subject = []
for nifti_path in glob.glob(labels_regex):
    nifti = NIfTI(nifti_path)
    total_dicoms_per_subject.append(nifti.shape[2])
    nifti.clean_3d_ndarray(inplace=True)
    cleaned_dicoms_per_subject.append(nifti.shape[2])
print(f'CLEANED DICOMS PER SUBJECT: MIN {min(cleaned_dicoms_per_subject)}, MAX {max(cleaned_dicoms_per_subject)}')

##
# creating NIfTI labels with their respective NIfTI images
n = NIfTI('label0001.nii.gz')
print(n.ndarray.shape)
# fisrt we need to select the are containing all the labels
_, selected_dat_idx = n.clean_3d_ndarray(height=96, inplace=True)
print(n.ndarray.shape)
# then we apply the resize over the area of interest (ROI)
n.resize((368, 368, 96), inplace=True)
print(n.ndarray.shape)
n.plot_3d_ndarray()
print(n.ndarray.shape)
n.save_as('pro_label_0001.nii.gz')

selected_dat_idx.shape
selected_dat_idx.nonzero()
dataset_path = 'CT-82'
path_to_subject_DICOMs = os.path.join(dataset_path, 'manifest-1599750808610', 'Pancreas-CT')
for i in range(1, 2):
    folder_regex = os.path.join(path_to_subject_DICOMs, f'PANCREAS_{i:04d}', '**/*.dcm')
    selected_dicoms = np.array(glob.glob(folder_regex, recursive=True))[selected_dat_idx].tolist()
    ProNIfTI.create_save(
        selected_dicoms,
        processing={'resize': {'target': (256, 256)}}, saving_path=f'pro_CT_{i:02}.pro.nii.gz'
    )

##
dcm = DICOM('new_image.dcm')
plt.hist(dcm.ndarray.flatten())
plt.show()
img = dcm.equalize_histogram(saving_path='new_image.png')
plt.hist(img.flatten())
plt.show()
img = dcm.equalize_histogram(clahe=True, saving_path='new_image_clahe.png')
plt.hist(img.flatten())
plt.show()
##

###############################################################################
#                                CT-82 dataset                                #
###############################################################################
# DICOM files 18942
# NIfTI labels 82
# MIN_VAL = -2048
# MAX_VAL = 3071

# folders PANCREAS_0025 and PANCREAS_0070 are empty
# MIN DICOMS per subject 181
# MAX DICOMS per subject 466
# DICOMS with data (cleaned) per subject: MIN 46, MAX 145
###############################################################################
#                                     dpis                                    #
# https://www.iprintfromhome.com/mso/understandingdpi.pdf very good explanantion of dpi
# dpi = dim px / size inches
# width = height = 512px
# print size = 7.1111111... inches
# dpi = 72 / 25.4 = 2.8346 pixel/mm

# target = isotropic 2.00 pixel/mm  = 2*25.4 = 50.8 dpi or pixel/in
# print size = 7.111111.. inches
# widh = height = 50.8 * 7.11111 = 361.2443 px

# 361.2443 % 16 != 0 so to avoid any padding or resize when using UNet
# we can use 352x352[49.50 dpi or 1.9488 px/in] or 368x368 [51.75 dpi 2.0374 px/in]

# using width = height = 160
# dpi = 22.500 = 0.8858 px/in

###############################################################################
###############################################################################
#                                 UNEt details                                #
# 3D model
# small batches 2 - 4
# standard data-augmentation techniques (affine transformations, axial flips, random crops)
# Intensity values are linearly scaled to obtain a normal distribution N (0, 1)
# Sorensen-Dice loss
# CT-150 train 120 testing 30
# The results on pancreas predictions demonstrate that attention gates (AGs)
# increase recall values (p = .005) by improving the model’s expression power as it relies
# on AGs to localise foreground pixels.
# inference timeused 160x160x96 tensors
# CT-80 (TCIA Pancreas-CT Dataset) train 61, test 21
# #+caption: models from scratch
# | Method          | Dice        | Precision   | Recall      | S2S dist(mm) |
# |-----------------+-------------+-------------+-------------+--------------|
# | U-Net [24]      | 0.815±0.068 | 0.815±0.105 | 0.826±0.062 | 2.576±1.180  |
# | Attention U-Net | 0.821±0.057 | 0.815±0.093 | 0.835±0.057 | 2.333±0.856  |
# model config https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/configs/config_unet_ct_dsv.json
# "augmentation": {
#     "acdc_sax": {
#       "shift": [0.1,0.1],
#       "rotate": 15.0,
#       "scale": [0.7,1.3],
#       "intensity": [1.0,1.0],
#       "random_flip_prob": 0.5,
#       "scale_size": [160,160,96],
#       "patch_size": [160,160,96]
#     }
#   },
###############################################################################
###############################################################################
#                                    DICOM                                    #
# https://towardsdatascience.com/understanding-dicoms-835cd2e57d0b
# 16 bit DICOM images have values ranging from -32768 to 32768 while 8-bit grey-scale images
# store values from 0 to 255.
###############################################################################

##
# Affine transformations examples: translation, scaling, homothety, similarity, reflection,
# rotation, shear mapping and compositions of them in any combination sequence
# torchvision transforms does not support 3D data
# - https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch#answer-68131471
#   fivecrop and tencrop can be used to augment the dataset number
# http://pytorch.org/vision/main/generated/torchvision.transforms.functional.affine.html
# https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html#torchvision.transforms.RandomAffine
# https://github.com/ncullen93/torchsample
#
# https://torchio.readthedocs.io/index.html https://github.com/fepegar/torchio
# https://github.com/Project-MONAI/MONAI/tree/dev/monai/transforms
# https://github.com/albumentations-team/albumentations/issues/138 albumentations does not work with 3d
# https://github.com/aleju/imgaug https://github.com/aleju/imgaug/issues/49
#
##
