import os
from os.path import join, basename, isdir, isfile
from collections import MutableMapping
from yaml import dump
from shutil import copytree, copyfile, rmtree


def load(path, h5_group=None):
    if h5_group is None:
        return load_path(path)
    else:
        return load_h5(path, h5_group)


def load_path(path):
    if isdir(path):
        return load_dir(path)
    elif isfile(path):
        return load_file(path)
    else:
        raise KeyError(path)


def load_dir(path):
    # if is_exp_sequence:
    #    return ExpSequence(path)
    return Directory(path)


def load_file(path):
    return File(path)


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


# noinspection PyTypeChecker
def dir_to_dict(path: str, include_files='all'):
    directory = {}

    for root, dirs, filenames in os.walk(path):
        dn = os.path.basename(root)
        directory[dn] = []

        if dirs:
            for d in dirs:
                directory[dn].append(dir_to_dict(path=join(path, d), include_files=include_files))

            if include_files is not 'all':
                len_filenames = len(filenames)
                filenames = filenames[:include_files]
                if len_filenames > include_files:
                    filenames.append(r'...')
            for f in filenames:
                directory[dn].append(f)
        else:
            if include_files is not 'all':
                len_filenames = len(filenames)
                filenames = filenames[:include_files]
                if len_filenames > include_files:
                    filenames.append(r'...')
            directory[dn] = filenames

        return directory


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

    def __init__(self, path):
        if not isdir(path):
            os.makedirs(path)
        self.path = path
        self.__name__ = basename(path)

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
        for key in os.listdir(self.path):
            path = join(self.path, key)
            yield path

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

    def tree(self, include_files='all'):
        print(dump(dir_to_dict(self.path, include_files)))

    def rmtree(self):
        rmtree(self.path)

    @GetterWithItem
    def iloc(self, index):
        key = list(self.keys())[index]
        return self[key]


class File:
    def __init__(self, path):
        self.path = path

    def open(self):
        raise NotImplementedError("File type is not yet implemented")

    def __repr__(self):
        return "file: " + self.path

    @property
    def file_type(self):
        file_name = basename(self.path)
        return file_name.split('.')[-1]
