# -*- coding: utf-8 -*-
""" nns/mixins/subdatasets """

from gtorch_utils.datasets.segmentation import DatasetTemplate
from torch.utils.data import DataLoader


class SubDatasetsMixin:
    """
    Provides methods to handle the subdatasets

    Usage:
        class SomeClass(SubDatasetsMixin):
            def __init__(self, **kwargs):
                # some lines of code
                self._SubDatasetsMixin__init(**kwargs)
    """

    def __init_and_validate(self, **kwargs):
        """
        Validates the arguments and sets the parameters as instance variables

        Kwargs:
            dataset (DatasetTemplate): Custom dataset class descendant of gtorch_utils.datasets.segmentation.DatasetTemplate.
                See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
            dataset_kwargs (dict): keyword arguments for the dataset. Default {}
            train_dataloader_kwargs <dict>: Keyword arguments for the train DataLoader.
                Default {'batch_size': 1, 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
            testval_dataloader_kwargs <dict>: Keyword arguments for the test and validation DataLoaders.
                Default {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True, 'drop_last': True}
        """
        self.dataset = kwargs['dataset']
        self.dataset_kwargs = kwargs.get('dataset_kwargs', {})
        self.train_dataloader_kwargs = kwargs.get(
            'train_dataloader_kwargs',
            {'batch_size': 1, 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
        )
        self.testval_dataloader_kwargs = kwargs.get(
            'testval_dataloader_kwargs',
            {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True, 'drop_last': True}
        )

        assert issubclass(self.dataset, DatasetTemplate), type(self.dataset)
        assert isinstance(self.dataset_kwargs, dict), type(self.dataset_kwargs)
        assert isinstance(self.train_dataloader_kwargs, dict), type(self.train_dataloader_kwargs)
        assert isinstance(self.testval_dataloader_kwargs, dict), type(self.testval_dataloader_kwargs)

    def __get_and_set_subdatasets(self):
        """
        Gets the sub datasets and set their corresponding Dataloaders instances into
        instance attributes
        """
        train, val, test = self.dataset.get_subdatasets(**self.dataset_kwargs)
        self.n_train = len(train) if train is not None else 0
        self.n_val = len(val) if val is not None else 0
        self.n_test = len(test) if test is not None else 0
        self.train_loader = DataLoader(train, **self.train_dataloader_kwargs) if train is not None else None
        self.val_loader = DataLoader(val, **self.testval_dataloader_kwargs) if val is not None else None
        self.test_loader = DataLoader(test, **self.testval_dataloader_kwargs) if test is not None else None

    def __init(self, **kwargs):
        """
        Validates the arguments and creates instance attributes containing the Dataloaders and lenghts
        of the subdatasets
        """
        self.__init_and_validate(**kwargs)
        self.__get_and_set_subdatasets()
