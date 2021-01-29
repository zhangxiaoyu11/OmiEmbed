"""
This module implements an abstract base class for datasets. Other datasets can be created from this base class.
"""
import torch.utils.data as data
from abc import ABC, abstractmethod


class BasicDataset(data.Dataset, ABC):
    """
    This class is an abstract base class for datasets.
    To create a subclass, you need to implement the following three functions:
    -- <__init__>:                      initialize the class, first call BasicDataset.__init__(self, param).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """

    def __init__(self, param):
        """
        Initialize the class, save the parameters in the class
        """
        self.param = param
        self.sample_list = None

    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        Parameters:
            index - - a integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        pass
