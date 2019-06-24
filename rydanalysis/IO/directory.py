import os
from os.path import join, basename, isdir, isfile
from collections import Mapping
from yaml import dump


def dir_to_dict(path, include_files=False):
    directory = {}

    for name, names, filenames in os.walk(path):
        dn = os.path.basename(name)
        directory[dn] = []

        if names:
            for d in names:
                directory[dn].append(dir_to_dict(os.path.join(path, d), include_files))

            if include_files:
                for f in filenames:
                    directory[dn].append(f)
        elif include_files:
            directory[dn] = filenames

        return directory


class Directory(Mapping):

    def __init__(self, path, *args, **kwargs):
        self.path = path
        self.__name__ = basename(path)

    def __getitem__(self, key):
        folder = join(self.path, key)
        if os.path.isdir(folder):
            return Directory(folder)
        else:
            return folder

    def __iter__(self):
        for key in os.listdir(self.path):
            folder = join(self.path, key)
            if isdir(folder):
                yield Directory(folder)

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
