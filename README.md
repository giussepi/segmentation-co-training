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
        "img": (".png", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Test/Images/"),
        "ann": (".mat", "/home/giussepi/Public/datasets/segmentation/consep/CoNSeP/Test/Labels/"),
    },
}

ProcessDataset(dataset_info=db_info)()
```

## LOGGING
This application is using [logzero](https://logzero.readthedocs.io/en/latest/). Thus, some functionalities can print extra data. To enable this just open your `settings.py` and set `DEBUG = True`. By default, the log level is set to [logging.INFO](https://docs.python.org/2/library/logging.html#logging-levels).


## TODO

- [x] ...
