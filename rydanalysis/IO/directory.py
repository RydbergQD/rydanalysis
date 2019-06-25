import os
from os.path import join, basename, isdir, isfile
from collections import MutableMapping
from yaml import dump
from shutil import copytree, copyfile, rmtree


def dir_to_dict(path: str, include_files=False):
    directory = {}

    for root, dirs, filenames in os.walk(path):
        dn = os.path.basename(root)
        directory[dn] = []

        if dirs:
            for d in dirs:
                # noinspection PyTypeChecker
                directory[dn].append(dir_to_dict(path=join(path, d), include_files=include_files))

            if include_files:
                for f in filenames:
                    directory[dn].append(f)
        elif include_files:
            directory[dn] = filenames

        return directory


class Directory(MutableMapping):

    def __init__(self, path):
        if not isdir(path):
            os.makedirs(path)
        self.path = path
        self.__name__ = basename(path)

    def __getitem__(self, key):
        path = join(self.path, key)
        if os.path.isdir(path):
            return Directory(path)
        elif os.path.isfile(path):
            return File(path)
        else:
            raise KeyError(key)

    def __setitem__(self, key: str, file_or_dir):
        if isinstance(file_or_dir, Directory):
            copytree(file_or_dir.path, join(self.path, key))
        elif key in self:
            raise FileExistsError("[WinError 183] Cannot create a file when that file already exists: "
                                  + self[key].path)
        else:
            copyfile(file_or_dir.path, join(self.path, key))

    def __delitem__(self, key):
        folder = join(self.path, key)
        if os.path.isdir(folder):
            return rmtree(folder)
        else:
            return os.remove(folder)

    def __iter__(self):
        for key in os.listdir(self.path):
            folder = join(self.path, key)
            if isdir(folder):
                yield folder

    def iter_files(self):
        for key in os.listdir(self.path):
            folder = join(self.path, key)
            if isfile(folder):
                yield folder

    def __len__(self):
        _len = 0
        for key in os.listdir(self.path):
            folder = join(self.path, key)
            if isdir(folder):
                _len += 1
        return _len

    def __repr__(self):
        return "directory: " + self.path

    def structure(self, include_files=False):
        print(dump(dir_to_dict(self.path, include_files)))


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
