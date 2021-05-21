"""
This package about data loading and data preprocessing
"""
import os
import torch
import importlib
import numpy as np
import pandas as pd
from util import util
from datasets.basic_dataset import BasicDataset
from datasets.dataloader_prefetch import DataLoaderPrefetch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def find_dataset_using_name(dataset_mode):
    """
    Get the dataset of certain mode
    """
    dataset_filename = "datasets." + dataset_mode + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # Instantiate the dataset class
    dataset = None
    # Change the name format to corresponding class name
    target_dataset_name = dataset_mode.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BasicDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BasicDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def create_dataset(param):
    """
    Create a dataset given the parameters.
    """
    dataset_class = find_dataset_using_name(param.omics_mode)
    # Get an instance of this dataset class
    dataset = dataset_class(param)
    print("Dataset [%s] was created" % type(dataset).__name__)

    return dataset


class CustomDataLoader:
    """
    Create a dataloader for certain dataset.
    """
    def __init__(self, dataset, param, shuffle=True, enable_drop_last=False):
        self.dataset = dataset
        self.param = param

        drop_last = False
        if enable_drop_last:
            if len(dataset) % param.batch_size < 3*len(param.gpu_ids):
                drop_last = True

        # Create dataloader for this dataset
        self.dataloader = DataLoaderPrefetch(
            dataset,
            batch_size=param.batch_size,
            shuffle=shuffle,
            num_workers=int(param.num_threads),
            drop_last=drop_last,
            pin_memory=param.set_pin_memory
        )

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

    def get_A_dim(self):
        """Return the dimension of first input omics data type"""
        return self.dataset.A_dim

    def get_B_dim(self):
        """Return the dimension of second input omics data type"""
        return self.dataset.B_dim

    def get_omics_dims(self):
        """Return a list of omics dimensions"""
        return self.dataset.omics_dims

    def get_class_num(self):
        """Return the number of classes for the downstream classification task"""
        return self.dataset.class_num

    def get_values_max(self):
        """Return the maximum target value of the dataset"""
        return self.dataset.values_max

    def get_values_min(self):
        """Return the minimum target value of the dataset"""
        return self.dataset.values_min

    def get_survival_T_max(self):
        """Return the maximum T of the dataset"""
        return self.dataset.survival_T_max

    def get_survival_T_min(self):
        """Return the minimum T of the dataset"""
        return self.dataset.survival_T_min

    def get_sample_list(self):
        """Return the sample list of the dataset"""
        return self.dataset.sample_list


def create_single_dataloader(param, shuffle=True, enable_drop_last=False):
    """
    Create a single dataloader
    """
    dataset = create_dataset(param)
    dataloader = CustomDataLoader(dataset, param, shuffle=shuffle, enable_drop_last=enable_drop_last)
    sample_list = dataset.sample_list

    return dataloader, sample_list


def create_separate_dataloader(param):
    """
    Create set of dataloader (train, val, test).
    """
    full_dataset = create_dataset(param)
    full_size = len(full_dataset)
    full_idx = np.arange(full_size)

    if param.not_stratified:
        train_idx, test_idx = train_test_split(full_idx,
                                               test_size=param.test_ratio,
                                               train_size=param.train_ratio,
                                               shuffle=True)
    else:
        if param.downstream_task == 'classification':
            targets = full_dataset.labels_array
        elif param.downstream_task == 'survival':
            targets = full_dataset.survival_E_array
            if param.stratify_label:
                targets = full_dataset.labels_array
        elif param.downstream_task == 'multitask':
            targets = full_dataset.labels_array
        elif param.downstream_task == 'alltask':
            targets = full_dataset.labels_array[0]
        train_idx, test_idx = train_test_split(full_idx,
                                               test_size=param.test_ratio,
                                               train_size=param.train_ratio,
                                               shuffle=True,
                                               stratify=targets)

    val_idx = list(set(full_idx) - set(train_idx) - set(test_idx))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    full_dataloader = CustomDataLoader(full_dataset, param)
    train_dataloader = CustomDataLoader(train_dataset, param, enable_drop_last=True)
    val_dataloader = CustomDataLoader(val_dataset, param, shuffle=False)
    test_dataloader = CustomDataLoader(test_dataset, param, shuffle=False)

    return full_dataloader, train_dataloader, val_dataloader, test_dataloader


def load_file(param, file_name):
    """
    Load data according to the format.
    """
    if param.file_format == 'tsv':
        file_path = os.path.join(param.data_root, file_name + '.tsv')
        print('Loading data from ' + file_path)
        df = pd.read_csv(file_path, sep='\t', header=0, index_col=0, na_filter=param.detect_na)
    elif param.file_format == 'csv':
        file_path = os.path.join(param.data_root, file_name + '.csv')
        print('Loading data from ' + file_path)
        df = pd.read_csv(file_path, header=0, index_col=0, na_filter=param.detect_na)
    elif param.file_format == 'hdf':
        file_path = os.path.join(param.data_root, file_name + '.h5')
        print('Loading data from ' + file_path)
        df = pd.read_hdf(file_path, header=0, index_col=0)
    else:
        raise NotImplementedError('File format %s is supported' % param.file_format)
    return df


def get_survival_y_true(param, T, E):
    """
    Get y_true for survival prediction based on T and E
    """
    # Get T_max
    if param.survival_T_max == -1:
        T_max = T.max()
    else:
        T_max = param.survival_T_max

    # Get time points
    time_points = util.get_time_points(T_max, param.time_num)

    # Get the y_true
    y_true = []
    for i, (t, e) in enumerate(zip(T, E)):
        y_true_i = np.zeros(param.time_num + 1)
        dist_to_time_points = [abs(t - point) for point in time_points[:-1]]
        time_index = np.argmin(dist_to_time_points)
        # if this is a uncensored data point
        if e == 1:
            y_true_i[time_index] = 1
            y_true.append(y_true_i)
        # if this is a censored data point
        else:
            y_true_i[time_index:] = 1
            y_true.append(y_true_i)
    y_true = torch.Tensor(y_true)

    return y_true
