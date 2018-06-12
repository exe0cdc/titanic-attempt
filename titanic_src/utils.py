import datetime
import sys
from pathlib import Path

import os.path as path
import pandas as pd
import numpy as np
import scipy.interpolate as interp
from sklearn.externals import joblib

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
LOG_PATH = path.join(PROJECT_DIR, 'log.log')


def get_time_now():
    """Gets current time in nice string format."""
    return str(datetime.datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]


def unclick(f):
    """aliases function as _function without click's decorators."""
    setattr(sys.modules[f.__module__], '_' + f.__name__, f)
    return f


def abs_path(file_path):
    """Returns a path with the project dir path prepended."""
    return path.join(PROJECT_DIR, file_path)


def get_log_filepath():
    """Returns the path for the log file."""
    return LOG_PATH


def get_project_dir():
    """Returns the main project path."""
    return PROJECT_DIR


def read_csv(file_path, *args, **kwargs):
    """Reads a csv file to a pandas dataframe from a path within the project dir."""
    df = pd.read_csv(abs_path(file_path), *args, **kwargs)
    return df


def read_pkl(file_path, *args, **kwargs):
    """Reads a pickled pandas dataframe from a path within the project dir."""
    df = pd.read_pickle(abs_path(file_path), *args, **kwargs)
    return df


def save_csv(df, file_path, *args, **kwargs):
    """Save a pandas dataframe to a csv within the project dir."""
    df.to_csv(abs_path(file_path), *args, **kwargs)


def save_pkl(df, file_path, *args, **kwargs):
    """Save a pandas dataframe to a pickle within the project dir."""
    df.to_pickle(abs_path(file_path), *args, **kwargs)


def save_pipeline(pipeline, file_path, *args, **kwargs):
    """Save a pipeline to a path within the project dir."""
    joblib.dump(pipeline, abs_path(file_path), *args, **kwargs)


def load_pipeline(file_path, *args, **kwargs):
    """Load a pipeline from a path within the project dir."""
    pipeline = joblib.load(abs_path(file_path), *args, **kwargs)
    return pipeline


def get_last_two(string):
    """Shortens a long pipeline description string to show only the most
       relevant information.

       Used in constructing a report df for model tester.
    """
    split_string = string.split('__')
    if len(split_string) > 2:
        new_str = '...{}__{}'.format(*split_string[-2:])
    else:
        new_str = string
    return new_str


def interpolate_array(array, new_len):
    """Shrinks or lengthens an array to a new length.

    Interpolates when more data points are needed.

    :param array: Original array
    :param new_len: len of new array
    :return: new array
    """
    array_interp = interp.interp1d(np.arange(array.size), array)
    array_compress = array_interp(np.linspace(0, array.size - 1, new_len))
    return array_compress
