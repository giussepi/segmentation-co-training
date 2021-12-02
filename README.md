# segmentation-co-training

## Installation

1. Clone this repository

2. [OPTIONAL] Create your virtual enviroment and activate it

3. Install [OpenSlide](https://openslide.org/download/)

4. Install the requirements

   `pip install -r requirements`

5. Make a copy of the configuration file and update it properly
   `cp settings.py.template settings.py`


## Usage

The main rule is running everything from the `main.py`


## Main Features

**Note**: Always see the class or function definition to pass the correct parameters and see all available options.

### Extract patches from [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

```python
from consep.utils.patches.patches import ProcessDataset

patch_size = (540, 540)
step_size = (164, 164)


db_info = {
    "train": {
        "img": (".png", "<some_path>/CoNSeP/Train/Images/"),
        "ann": (".mat", "<some_path>/CoNSeP/Train/Labels/"),
    },
    "valid": {
        "img": (".png", "<some_path>/CoNSeP/Test/Images/"),
        "ann": (".mat", "<some_path>/CoNSeP/Test/Labels/"),
    },
}

ProcessDataset(
    dataset_info=db_info, win_size=patch_size, step_size=step_size, ann_percentage=0.3)()
```

### Loading patches from [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

```python
import glob
import os

import torch
from gtorch_utils.constants import DB
from torch.utils.data import DataLoader

from consep.dataloaders import OnlineCoNSePDataset, SeedWorker

num_gpus = 1
model_input_shape = (270, 270)
model_outut_shape = (270, 270) # (80, 80)
batch_size = 16  # train and val
run_mode = DB.TRAIN
num_workers = 16

train_path = 'dataset/training_data/consep/train/540x540_164x164'
train_list = glob.glob(os.path.join(train_path, '*.npy'))
train_list.sort()

# val_path = 'dataset/training_data/consep/valid/540x540_164x164'

input_dataset = OnlineCoNSePDataset(
    file_list=train_list,
    input_shape=model_input_shape,
    mask_shape=model_outut_shape,
    mode=DB.TRAIN,
    setup_augmentor=True,
)

train_dataloader = DataLoader(
    input_dataset,
    num_workers=num_workers,
    batch_size=batch_size * num_gpus,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=False)(),
)

# test the loader
data = next(iter(train_dataloader))

for i in range(batch_size * num_gpus):
    plot_img_and_mask(data['img'][i, :], data['mask'][i, :])
```

### Creating and loading crop dataset from patches (offline data augmentation)
```python
from torch.utils.data import DataLoader

from consep.processors.offline import CreateDataset
from consep.dataloaders import SeedWorker, OfflineCoNSePDataset


CreateDataset(
    train_path='<path_to_consep_train_subdataset>',
    val_path='<path_to_consep_validation_subdataset>',
)()

train, val, test = OfflineCoNSePDataset.get_subdatasets(
    train_path='consep_dataset/train', val_path='consep_dataset/val')

train_dataloader = DataLoader(
    train,
    num_workers=0,
    batch_size=batch_size * num_gpus,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=True)(),
)
data = next(iter(train_dataloader))
```

## LOGGING
This application is using [logzero](https://logzero.readthedocs.io/en/latest/). Thus, some functionalities can print extra data. To enable this just open your `settings.py` and set `DEBUG = True`. By default, the log level is set to [logging.INFO](https://docs.python.org/2/library/logging.html#logging-levels).


## TODO

- [x] ...
