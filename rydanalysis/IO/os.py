import os
from os.path import join, basename, isdir, isfile, splitext
from collections import MutableMapping, Iterable
from yaml import dump
from shutil import copytree, copyfile, rmtree
import h5py


def load(path, h5_group=None, lazy=False):
    if h5_group is None:
        return load_path(path, lazy=lazy)
    else:
        return load_h5(path, h5_group, lazy=lazy)


def load_path(path, lazy=False):
    if isdir(path):
        return load_dir(path)
    elif isfile(path):
        return load_file(path, lazy=lazy)
    else:
        raise KeyError(path)


def load_dir(path):
    # if is_exp_sequence:
    #    return ExpSequence(path)
    return Directory(path)


def load_file(path, lazy=False):
    file_extension = splitext(path)[1]
    if file_extension == '.h5':
        return H5File(path, lazy=lazy)
    return File(path)


def load_h5(path, h5_key, lazy=False):
    with h5py.File(path, 'r') as hf:
        if isinstance(hf[h5_key], h5py.Group):
            return H5Group(path, h5_key, lazy=lazy)
        elif isinstance(hf[h5_key], h5py.Dataset):
            if lazy:
                return H5Dataset(path, h5_key)
            else:
                return hf[h5_key][()]
        else:
            raise NotImplementedError()


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
    raise NotImplementedError()


def tree(origin, include_files='all'):
    tree_dict = {origin.__name__: []}
    origin.lazy = True

    for key, item in origin.items():
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


class Directory(MutableMapping):

    def __init__(self, path, lazy=False):
        if not isdir(path):
            os.makedirs(path)
        self.path = path
        self.__name__ = basename(path)
        self.lazy = lazy

    def __getitem__(self, key):
        path = join(self.path, key)
        return load_path(path)

    def __setitem__(self, key: str, file_or_dir):
        if isinstance(file_or_dir, Directory):
            copytree(file_or_dir.path, join(self.path, key))
        elif key in self:
            raise FileExistsError("[WinError 183] Cannot create a file when that file already exists: "
                                  + self[key].path)
        else:
            copyfile(file_or_dir.path, join(self.path, key))

    def __delitem__(self, key):
        path = join(self.path, key)
        remove_path(path)

    def __iter__(self):
        return iter(os.listdir(self.path))

    def _ipython_key_completions_(self):
        return os.listdir(self.path)

    def iter_dirs(self):
        for key in os.listdir(self.path):
            path = join(self.path, key)
            if isdir(path):
                yield path

    def iter_files(self):
        for key in os.listdir(self.path):
            path = join(self.path, key)
            if isfile(path):
                yield path

    def __len__(self):
        _len = 0
        for key in os.listdir(self.path):
            path = join(self.path, key)
            if isdir(path):
                _len += 1
        return _len

    def __repr__(self):
        return "directory: " + self.path

    def __str__(self):
        return "directory: " + self.__name__

    def tree(self, include_files='all'):
        print(dump(tree(self, include_files)))

    def rmtree(self):
        rmtree(self.path)

    @GetterWithItem
    def iloc(self, index):
        key = list(self.keys())[index]
        return self[key]


class File:
    def __init__(self, path):
        self.path = path
        self.__name__ = basename(path)

    def open(self):
        raise NotImplementedError("File type is not yet implemented")

    def __repr__(self):
        return "file: " + self.path

    def __str__(self):
        return "file: " + self.__name__

    @property
    def file_type(self):
        file_name = basename(self.path)
        return file_name.split('.')[-1]


def h5_join(path, *paths):
    return join(path, *paths).replace('\\', '/')


class H5File(MutableMapping):

    def __init__(self, path, lazy=False):
        if not isfile(path):
            with h5py.File(join(path, 'test2.h5'), 'w') as _:
                pass
        self.path = path
        self.__name__ = basename(path)
        self.lazy = lazy

    def __getitem__(self, h5_key):
        return load_h5(self.path, h5_key, lazy=self.lazy)

    def __setitem__(self, key: str, data):
        with h5py.File(self.path, 'r+') as hf:
            hf[key] = data

    def __delitem__(self, key):
        with h5py.File(self.path, 'r+') as hf:
            del hf[key]

    def __iter__(self):
        with h5py.File(self.path, 'r+') as hf:
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

    def tree(self, include_files='all'):
        return print(dump(tree(self, include_files)))

    def remove(self):
        os.remove(self.path)

    def create_group(self, name, track_order=None):
        with h5py.File(self.path, 'r+') as hf:
            hf.create_group(name, track_order)
        return H5Group(self.path, name)


class H5Group(MutableMapping):
    def __init__(self, path, h5_key, lazy=False):
        self.path = path
        self.h5_key = h5_key
        self.__name__ = basename(h5_key)
        self.lazy = lazy

    def __getitem__(self, key):
        h5_key = h5_join(self.h5_key, key)
        return load_h5(self.path, h5_key, self.lazy)

    def __setitem__(self, key: str, data):
        h5_key = h5_join(self.h5_key, key)
        with h5py.File(self.path, 'r+') as hf:
            hf[h5_key] = data

    def __delitem__(self, key):
        h5_key = h5_join(self.h5_key, key)
        with h5py.File(self.path, 'r+') as hf:
            del hf[h5_key]

    def __iter__(self):
        with h5py.File(self.path, 'r+') as hf:
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
        with h5py.File(self.path) as hf:
            return str(hf[self.h5_key])
