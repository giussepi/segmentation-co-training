# TCIA Pancreas CT-82 [^1][^2][^3]
## Getting the data
1. Download it from [https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#2251404047e506d05d9b43829c2200c8c77afe3b](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#2251404047e506d05d9b43829c2200c8c77afe3b)

2. Move it inside the main directory of this project

3. Rename it to `CT-82`

## Process the dataset

``` python
import glob
import os
from ct82.images import NIfTI, ProNIfTI
from ct82.processors import CT82MGR

target_size = (368, 368, 96)
mgr = CT82MGR(target_size=target_size)
mgr()

assert len(glob.glob(os.path.join(mgr.saving_labels_folder, r'*.nii.gz'))) == 80
assert len(glob.glob(os.path.join(mgr.saving_cts_folder, r'*.pro.nii.gz'))) == 80

files_idx = [*range(1, 83)]
    for id_ in mgr.non_existing_ct_folders[::-1]:
        files_idx.pop(id_-1)

    for subject in files_idx:
        labels = NIfTI(os.path.join(mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
        cts = ProNIfTI(os.path.join(mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
        assert labels.shape == cts.shape == target_size
```

## Perform visual verification
To see the changes open visual_verification.png (it is created right after you execute the folliwing lines) and it will be continuosly updated with new mask data.

``` python
from ct82.processors import CT82MGR

target_size = (368, 368, 96)
mgr = CT82MGR(
    saving_path='CT-82-Pro',
    target_size=target_size
)
mgr.non_existing_ct_folders = []
mgr.perform_visual_verification(1, scans=[72], clahe=True)
# os.remove(mgr.VERIFICATION_IMG)
```

## Split into train, validation, test
```python
from ct82.processors import CT82MGR

target_size = (368, 368, 96)  # fixed num of scans/depth
target_size = (368, 368, -1)  # different depth per subject containing only scans with data

mgr = CT82MGR(
    saving_path='CT-82-Pro',
    target_size=target_size
)
mgr.split_processed_dataset(.15, .2, shuffle=False)
```

## Load and verify train, val and test subdatasets
``` python
import matplotlib.pyplot as plt
import numpy as np
from ct82.datasets import CT82Dataset

train, val, test = CT82Dataset.get_subdatasets(
    train_path='CT-82-Pro/train', val_path='CT-82-Pro/val', test_path='CT-82-Pro/test'
)
for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    print(f'{db_name}: {len(dataset)}')
    data = dataset[0]
    print(data['image'].shape, data['mask'].shape)
    print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])

    print(data['image'].min(), data['image'].max())
    print(data['mask'].min(), data['mask'].max())

    img_id = np.random.randint(0, 72)
    if len(data['image'].shape) == 4:
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(data['image'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
        axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
        plt.show()
    else:
        fig, axis = plt.subplots(2, 4)
        for idx, i, m in zip([*range(4)], data['image'], data['mask']):
            axis[0, idx].imshow(i.detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
            axis[1, idx].imshow(m.detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
        plt.show()
```


[^1]: Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
[^2]: Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.  ([paper](http://arxiv.org/pdf/1506.06448.pdf))
[^3]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)
