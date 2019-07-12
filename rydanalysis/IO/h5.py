from os.path import isfile, join, basename
from collections import MutableMapping
from h5py import File, Dataset, Group
import os


def h5_join(path, *paths):
    return join(path, *paths).replace('\\', '/')


def load_h5(path, h5_key):
    with File(path, 'r') as hf:
        if isinstance(hf[h5_key], Group):
            return H5Group(path, h5_key)
        elif isinstance(hf[h5_key], Dataset):
            return hf[h5_key][()]
        else:
            raise NotImplementedError()


class H5File(MutableMapping):

    def __init__(self, path):
        if not isfile(path):
            with File(join(path, 'test2.h5'), 'w') as _:
                pass
        self.path = path
        self.__name__ = basename(path)

    def __getitem__(self, h5_key):
        return load_h5(self.path, h5_key)

    def __setitem__(self, key: str, data):
        with File(self.path, 'r+') as hf:
            hf[key] = data

    def __delitem__(self, key):
        with File(self.path, 'r+') as hf:
            del hf[key]

    def __iter__(self):
        with File(self.path, 'r+') as hf:
            keys = list(hf.keys())
        return iter(keys)

    def _ipython_key_completions_(self):
        with File(self.path, 'r') as hf:
            return list(hf.keys())

    def __len__(self):
        with File(self.path, 'r') as hf:
            return len(hf)

    def __repr__(self):
        return "H5File: " + self.path

    def tree(self, include_files='all'):
        raise NotImplementedError()

    def remove(self):
        os.remove(self.path)

    def create_group(self, name, track_order=None):
        with File(self.path, 'r+') as hf:
            hf.create_group(name, track_order)
        return H5Group(self.path, name)


class H5Group(MutableMapping):
    def __init__(self, path, h5_key):
        self.path = path
        self.h5_key = h5_key

    def __getitem__(self, key):
        h5_key = h5_join(self.h5_key, key)
        return load_h5(self.path, h5_key)

    def __setitem__(self, key: str, data):
        h5_key = h5_join(self.h5_key, key)
        with File(self.path, 'r+') as hf:
            hf[h5_key] = data

    def __delitem__(self, key):
        h5_key = h5_join(self.h5_key, key)
        with File(self.path, 'r+') as hf:
            del hf[h5_key]

    def __iter__(self):
        with File(self.path, 'r+') as hf:
            keys = list(hf[self.h5_key].keys())
        return iter(keys)

    def _ipython_key_completions_(self):
        with File(self.path, 'r') as hf:
            return list(hf[self.h5_key].keys())

    def __len__(self):
        with File(self.path, 'r') as hf:
            return len(hf[self.h5_key])

    def __repr__(self):
        return "H5Group: " + self.path + 'grp: ' + self.h5_key

    def tree(self, include_files='all'):
        raise NotImplementedError()

    def remove(self):
        with File(self.path, 'r') as hf:
            del hf[self.h5_key]

    def create_group(self, name, track_order=None):
        with File(self.path, 'r+') as hf:
            hf[self.h5_key].create_group(name, track_order)
        return H5Group(self.path, h5_join(self.h5_key, name))


class H5Dataset:
    def __init__(self, path, h5_key):
        self.path = path
        self.h5_key = h5_key
