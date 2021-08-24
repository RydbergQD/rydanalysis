import os
from os.path import join, isdir, isfile, splitext, basename
from collections.abc import Iterable
from shutil import rmtree
import pandas as pd
import h5py
import rydanalysis


def h5_join(path, *paths):
    return join(path, *paths).replace('\\', '/')


def _load_helper(name, lazy=False, **kwargs):
    _class = getattr(rydanalysis, name)
    instance = _class(**kwargs)
    if not lazy:
        try:
            return instance.read()
        except AttributeError:
            pass
    return instance


def load(path, h5_group=None, lazy=False):
    if h5_group is None:
        return _load_path(path, lazy)
    else:
        return _load_h5(path, h5_group, lazy)


def _load_path(path, lazy=False):
    if isdir(path):
        return _load_dir(path, lazy=lazy)
    elif isfile(path):
        return _load_file(path, lazy=lazy)
    else:
        raise KeyError(path)


def _load_dir(path, lazy=False):
    if {'analysis', 'raw_data'} <= set(os.listdir(path)):
        return _load_helper("ExpSequence", path=path, lazy=lazy)
    if basename(path) == 'raw_data':
        return _load_helper('RawData', path=path, lazy=lazy)
    return _load_helper("Directory", path=path, lazy=lazy)


def _load_file(path, lazy=False):
    file_extension = splitext(path)[1]
    if file_extension == '.h5':
        ryd_type = get_ryd_type(path, '/')
        if ryd_type == 'single_shot':
            return _load_helper("SingleShot", path=path, lazy=lazy)
        return _load_helper("H5File", lazy=lazy, path=path)
    if file_extension == '.nc':
        return _load_helper("NetCDFFile", lazy=lazy,path=path)
    if file_extension in ['.fits', '.fts', 'fit']:
        return _load_helper("FitsFile", lazy=lazy, path=path)
    if file_extension in ['.csv', '.txt']:
        return _load_helper("CSVFile", lazy=lazy, path=path)
    return _load_helper("File", lazy=lazy, path=path)


def _load_h5(path, h5_key, lazy=False):
    # return pandas
    if test_pandas(path, h5_key):
        return _load_helper("PandasDataset", path=path, h5_key=h5_key, lazy=lazy)

    # return H5Group or H5Dataset or Data
    with h5py.File(path, 'r') as hf:
        if isinstance(hf[h5_key], h5py.Group):
            return _load_helper("H5Group", path=path, h5_key=h5_key, lazy=lazy)
        elif isinstance(hf[h5_key], h5py.Dataset):
            return _load_helper("H5Dataset", path=path, h5_key=h5_key, lazy=lazy)
        else:
            raise NotImplementedError()


def load_fits(path, fits_index, lazy=False):
    return _load_helper("FitsDataset", path=path, fits_index=fits_index, lazy=lazy)


def test_pandas(path, h5_key):
    with h5py.File(path, 'r') as hf:
        return 'pandas_version' in hf[h5_key].attrs


def get_ryd_type(path, h5_key):
    with h5py.File(path, 'r') as hf:
        try:
            return hf[h5_key].attrs['ryd_type']
        except KeyError:
            return False


def remove(path, h5_group=None):
    if h5_group is None:
        return remove_path(path)
    else:
        return remove_h5(path, h5_group)


def remove_path(path):
    if isdir(path):
        return os.rmdir(path)
    elif isfile(path):
        return os.remove(path)
    else:
        raise KeyError(path)


def remove_h5(path, h5_group):
    raise NotImplementedError("Remove {} in {} not implemented".format(h5_group, path))


def tree(origin, include_files='all'):
    tree_dict = {origin.__name__: []}
    origin.lazy = True

    for key in origin.keys():
        item = origin.lazy_get(key)
        if isinstance(item, Iterable):
            # noinspection PyTypeChecker
            tree_dict[origin.__name__].append(tree(item, include_files=include_files))
        else:
            tree_dict[origin.__name__].append(str(item))
    return tree_dict


class GetterWithItem:
    def __init__(self, f_get_item):
        self.f_get_item = f_get_item

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return WithItem(self.f_get_item, instance)


class WithItem:
    def __init__(self, fct, instance):
        self.fct = fct
        self.instance = instance

    def __getitem__(self, item):
        return self.fct(self.instance, item)

    def __delitem__(self, key):
        folder = join(self.instance.path, list(self.instance.keys())[key])
        if os.path.isdir(folder):
            return rmtree(folder)
        else:
            return os.remove(folder)


def str_to_datetime(time_str: str):
    """Transforms str in format '%Y_%m_%d_%H.%M.%S' to pd.Datetime"""
    return pd.to_datetime(time_str, format='%Y_%m_%d_%H.%M.%S')


def datetime_to_str(tmstp: pd.Timestamp):
    """Transforms pd.Datetime to str in format '%Y_%m_%d_%H.%M.%S'"""
    return tmstp.strftime('%Y_%m_%d_%H.%M.%S')


class GetterWithTimeString(GetterWithItem):
    """Getter for class with tmstps. """
    def __get__(self, instance, owner):
        return WithTimeString(self.f_get_item, instance)


class WithTimeString(WithItem):
    def _ipython_key_completions_(self):
        tmstps = self.instance.instance.tmstps
        return [datetime_to_str(tmstp) for tmstp in tmstps]


class GetterWithTimestamp(GetterWithItem):
    def __get__(self, instance, owner):
        return WithTimestamp(self.f_get_item, instance)


class WithTimestamp(WithItem):
    @GetterWithTimeString
    def from_str(self, time_str: str):
        return self.__getitem__(str_to_datetime(time_str))
