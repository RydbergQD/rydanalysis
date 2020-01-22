from os.path import isfile, basename
from collections.abc import MutableMapping
import h5py
from yaml import dump
import os
import pandas as pd
import xarray as xr

from rydanalysis.IO.io import h5_join, load, tree


class H5File(MutableMapping):

    def __init__(self, path):
        if not isfile(path):
            with h5py.File(path, 'w') as _:
                pass
        self.path = path
        self.__name__ = basename(path)

    def __getitem__(self, h5_key):
        return load(self.path, h5_key)

    def __setitem__(self, key: str, data):
        if data is None:
            pass
        elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
            data.to_netcdf(path=self.path)
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data.to_hdf(self.path, key, format='fixed')
        else:
            with h5py.File(self.path, 'r+') as hf:
                hf[key] = data

    def __delitem__(self, key):
        with h5py.File(self.path, 'r+') as hf:
            del hf[key]

    def __iter__(self):
        with h5py.File(self.path, 'r') as hf:
            keys = list(hf.keys())
        return iter(keys)

    def _ipython_key_completions_(self):
        with h5py.File(self.path, 'r') as hf:
            return list(hf.keys())

    def __len__(self):
        with h5py.File(self.path, 'r') as hf:
            return len(hf)

    def __repr__(self):
        return "H5File: " + self.path

    def __str__(self):
        return "H5File: " + self.__name__

    def lazy_get(self, h5_key):
        return load(self.path, h5_key, lazy=True)

    def tree(self, include_files='all'):
        print(dump(tree(self, include_files)))

    def remove(self):
        os.remove(self.path)

    def create_group(self, name, track_order=None):
        with h5py.File(self.path, 'r+') as hf:
            hf.create_group(name, track_order)
        return H5Group(self.path, name)


class H5Group(MutableMapping):
    def __init__(self, path, h5_key):
        self.path = path
        self.h5_key = h5_key
        self.__name__ = basename(h5_key)

    def __getitem__(self, key):
        h5_key = h5_join(self.h5_key, key)
        return load(self.path, h5_key)

    def __setitem__(self, key: str, data):
        h5_key = h5_join(self.h5_key, key)
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data.to_hdf(self.path, h5_key, format='fixed', mode='a')
        elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
            data.to_netcdf(path=self.path, group=h5_key, mode='a')
        with h5py.File(self.path, 'r+') as hf:
            hf[h5_key] = data

    def __delitem__(self, key):
        h5_key = h5_join(self.h5_key, key)
        with h5py.File(self.path, 'r+') as hf:
            del hf[h5_key]

    def __iter__(self):
        with h5py.File(self.path, 'r') as hf:
            keys = list(hf[self.h5_key].keys())
        return iter(keys)

    def _ipython_key_completions_(self):
        with h5py.File(self.path, 'r') as hf:
            return list(hf[self.h5_key].keys())

    def __len__(self):
        with h5py.File(self.path, 'r') as hf:
            return len(hf[self.h5_key])

    def __repr__(self):
        return "H5Group: " + self.path + 'grp: ' + self.h5_key

    def __str__(self):
        return "H5Group: " + self.__name__

    def lazy_get(self, key):
        h5_key = h5_join(self.h5_key, key)
        return load(self.path, h5_key, lazy=True)

    def tree(self, include_files='all'):
        return print(dump(tree(self, include_files)))

    def remove(self):
        with h5py.File(self.path, 'r') as hf:
            del hf[self.h5_key]

    def create_group(self, name, track_order=None):
        with h5py.File(self.path, 'r+') as hf:
            hf[self.h5_key].create_group(name, track_order)
        return H5Group(self.path, h5_join(self.h5_key, name))


class H5Dataset:
    def __init__(self, path, h5_key):
        self.path = path
        self.h5_key = h5_key
        self.__name__ = basename(h5_key)

    def __repr__(self):
        with h5py.File(self.path, 'r') as hf:
            return str(hf[self.h5_key])

    def read(self):
        with h5py.File(self.path, 'r') as hf:
            return hf[self.h5_key][()]


class PandasDataset:
    def __init__(self, path, h5_key):
        self.path = path
        self.h5_key = h5_key
        self.__name__ = basename(h5_key)

    def __repr__(self):
        with h5py.File(self.path, 'r') as hf:
            return 'Pandas Dataset: ' + str(hf[self.h5_key])

    def read(self):
        return pd.read_hdf(self.path, self.h5_key)
