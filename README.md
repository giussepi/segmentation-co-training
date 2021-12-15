# segmentation-co-training

## Installation

1. Clone this repository

2. [OPTIONAL] Create your virtual enviroment and activate it

3. Install [OpenSlide](https://openslide.org/download/)

4. Install the requirements

   `pip install -r requirements`

5. Install pytorch following the instructions provided on the page [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

6. Make a copy of the configuration file and update it properly
   `cp settings.py.template settings.py`


## Usage

The main rule is running everything from the `main.py`

### Running the TensorBoard

1. Make `run_tensorboard.sh` executable:

	`chmod +x run_tensorboard.sh`

2. Execute it

	`./run_tensorboard.sh`

3. Open your browser and go to [http://localhost:6006/](http://localhost:6006/)


### Debugging

Just open your `settings.py` and set `DEBUG = True`. This will set the log level to debug and your dataloader will not use workers so you can use `pdb.set_trace()` without any problem.

## Main Features

**Note**: Always see the class or function definition to pass the correct parameters and see all available options.

### Extract patches from [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

```python
import settings
from consep.utils.patches.patches import ProcessDataset

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

# using 30% of the annotations
ProcessDataset(dataset_info=db_info, win_size=settings.PATCH_SIZE,
               step_size=settings.PATCH_STEP_SIZE, extract_type=PatchExtractType.MIRROR,
               type_classification=True, ann_percentage=.3)()
```

### Loading patches from [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

```python
import glob
import os

import torch
from gtorch_utils.constants import DB
from torch.utils.data import DataLoader

import settings
from consep.dataloaders import OnlineCoNSePDataset, SeedWorker

run_mode = DB.TRAIN
train_path = 'dataset/training_data/consep/train/540x540_164x164'
train_list = glob.glob(os.path.join(train_path, '*.npy'))
train_list.sort()

input_dataset = OnlineCoNSePDataset(
    file_list=train_list,
    crop_img_shape=settings.CROP_IMG_SHAPE,
    crop_mask_shape=settings.CROP_MASK_SHAPE,
    mode=DB.TRAIN,
    setup_augmentor=True,
)

train_dataloader = DataLoader(
    input_dataset,
    num_workers=settings.NUM_WORKERS,
    batch_size=settings.TOTAL_BATCH_SIZE,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=True)(),
)

data = next(iter(train_dataloader))

for i in range(settings.TOTAL_BATCH_SIZE):
    plot_img_and_mask(data['img'][i, :], data['mask'][i, :])
```

### Creating and loading crop dataset from patches (offline data augmentation)
```python
from torch.utils.data import DataLoader

import settings
from consep.processors.offline import CreateDataset
from consep.dataloaders import SeedWorker, OfflineCoNSePDataset

CreateDataset(
    train_path='dataset/training_data/consep/train/540x540_164x164',
    val_path='dataset/training_data/consep/valid/540x540_164x164',
    crop_img_shape=settings.CROP_IMG_SHAPE,
    crop_mask_shape=settings.CROP_MASK_SHAPE,
    num_gpus=settings.NUM_GPUS,
    num_workers=settings.NUM_WORKERS,
    saving_path=settings.CREATEDATASET_SAVING_PATH,
)()

run_mode = DB.TRAIN
train, val, test = OfflineCoNSePDataset.get_subdatasets(
    train_path=settings.CONSEP_TRAIN_PATH, val_path=settings.CONSEP_VAL_PATH)

train_dataloader = DataLoader(
    train,
    num_workers=settings.NUM_WORKERS,
    batch_size=settings.TOTAL_BATCH_SIZE,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=True)(),
)

data = next(iter(train_dataloader))
```

### Training, Testing and Plotting on TensorBoard
Use the `ModelMGR` to train models and make predictions.

``` python
import logzero

import torch
from gtorch_utils.nns.models.segmentation import UNet
from gtorch_utils.segmentation import metrics

import numpy as np
import settings
from consep.dataloaders import OfflineCoNSePDataset
from consep.datasets.constants import BinaryCoNSeP
from nns.managers import ModelMGR
from nns.mixins.constants import LrShedulerTrack


logzero.loglevel(settings.LOG_LEVEL)

ModelMGR(
    model=torch.nn.DataParallel(UNet(n_channels=3, n_classes=1, bilinear=True)),
    # model=UNet(n_channels=3, n_classes=1, bilinear=True),
    cuda=True,
    epochs=20,
    intrain_val=2,
    optimizer=torch.optim.Adam,
    optimizer_kwargs=dict(lr=1e-3),
    labels_data=BinaryCoNSeP,
    dataset=OfflineCoNSePDataset,
    dataset_kwargs={
        'train_path': settings.CONSEP_TRAIN_PATH,
        'val_path': settings.CONSEP_VAL_PATH,
        'test_path': settings.CONSEP_TEST_PATH,
    },
    train_dataloader_kwargs={
        'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': True, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False
    },
    testval_dataloader_kwargs={
        'batch_size': settings.TOTAL_BATCH_SIZE, 'shuffle': False, 'num_workers': settings.NUM_WORKERS, 'pin_memory': False, 'drop_last': True
    },
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # torch.optim.lr_scheduler.StepLR,
    # TODO: the mode can change based on the quantity monitored
    # get inspiration from https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
    lr_scheduler_kwargs={'mode': 'min', 'patience': 2},  # {'step_size': 10, 'gamma': 0.1},
    lr_scheduler_track=LrShedulerTrack.LOSS,
    criterions=[
        torch.nn.BCEWithLogitsLoss()
        # torch.nn.CrossEntropyLoss()
    ],
    mask_threshold=0.5,
    metric=metrics.dice_coeff_metric,
    earlystopping_kwargs=dict(min_delta=1e-3, patience=np.inf, metric=True),
    checkpoint_interval=1,
    train_eval_chkpt=True,
    ini_checkpoint='',
    dir_checkpoints=settings.DIR_CHECKPOINTS,
    tensorboard=True,
    # TODO: there a bug that appeared once when plotting to disk after a long training
    # anyway I can always plot from the checkpoints :)
    plot_to_disk=False,
    plot_dir=settings.PLOT_DIRECTORY
)()
```

### Showing Logs Summary from a Checkpoint
Use the `ModelMGR.print_data_logger_summary` method to do it.

``` python
model = ModelMGR(<your settings>, ini_checkpoint='chkpt_X.pth.tar', dir_checkpoints=settings.DIR_CHECKPOINTS)
model.print_data_logger_summary()
```

The summary will be a table like this one

| key         | Validation   |   corresponding training value |
|-------------|--------------|--------------------------------|
| Best metric | 0.7495       |                         0.7863 |
| Min loss    | 0.2170       |                         0.1691 |
| Max LR      |              |                         0.001  |
| Min LR      |              |                         1e-07  |


## LOGGING
This application is using [logzero](https://logzero.readthedocs.io/en/latest/). Thus, some functionalities can print extra data. To enable this just open your `settings.py` and set `DEBUG = True`. By default, the log level is set to [logging.INFO](https://docs.python.org/2/library/logging.html#logging-levels).


## TODO

- [x] ...
