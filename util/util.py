"""
Contain some simple helper functions
"""
import os
import shutil
import torch
import random
import numpy as np


def mkdir(path):
    """
    Create a empty directory in the disk if it didn't exist

    Parameters:
        path(str) -- a directory path we would like to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def clear_dir(path):
    """
    delete all files in a path

    Parameters:
        path(str) -- a directory path that we would like to delete all files in it
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


def setup_seed(seed):
    """
    setup seed to make the experiments deterministic

    Parameters:
        seed(int) -- the random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_time_points(T_max, time_num, extra_time_percent=0.1):
    """
    Get time points for the MTLR model
    """
    # Get time points in the time axis
    time_points = np.linspace(0, T_max * (1 + extra_time_percent), time_num + 1)

    return time_points
