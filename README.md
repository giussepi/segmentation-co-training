# segmentation-co-training

## Installation

1. Clone this repository

2. [OPTIONAL] Create your virtual enviroment and activate it

3. Install the requirements

   `pip install -r requirements`

4. Make a copy of the configuration file and update it properly
   `cp settings.py.template settings.py`


## Usage

The main rule is running everything from the `main.py`


## Main Features

**Note**: Always see the class or function definition to pass the correct parameters and see all available options.

### Extract patches from [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

```python
from utils.patches.patches import ProcessDataset

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

ProcessDataset(dataset_info=db_info)()
```

### Loading patches from [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

```python
import glob
import os

import torch
from gtorch_utils.constants import DB
from torch.utils.data import DataLoader

from dataloaders.train_loader import FileLoader, SeedWorker


train_path = 'dataset/training_data/consep/train/540x540_164x164'
train_list = glob.glob(os.path.join(train_path, '*.npy'))
train_list.sort()

# val_path = 'dataset/training_data/consep/valid/540x540_164x164'

input_dataset = FileLoader(
    file_list=train_list,
    input_shape=(270, 270),
    mask_shape=(164, 164),
    mode=DB.TRAIN,
    setup_augmentor=True,
)

# when debbuging set num_workers = 0
train_dataloader = DataLoader(
    input_dataset,
    num_workers=16,
    batch_size=batch_size * num_gpus,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=False)(),
)

# test the loader
data = next(iter(train_dataloader))
```


## LOGGING
This application is using [logzero](https://logzero.readthedocs.io/en/latest/). Thus, some functionalities can print extra data. To enable this just open your `settings.py` and set `DEBUG = True`. By default, the log level is set to [logging.INFO](https://docs.python.org/2/library/logging.html#logging-levels).


## TODO

- [x] ...
