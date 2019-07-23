import os
from os.path import join, basename, isdir, isfile
from collections.abc import MutableMapping
from yaml import dump
from shutil import copytree, copyfile, rmtree

import rydanalysis.IO.h5
from rydanalysis.IO.io import GetterWithItem, load, remove_path


class Directory(MutableMapping):

    def __init__(self, path):
        if not isdir(path):
            os.makedirs(path)
        self.path = path
        self.__name__ = basename(path)

    def __getitem__(self, key):
        path = join(self.path, key)
        return load(path)

    def lazy_get(self, key):
        path = join(self.path, key)
        return load(path, lazy=True)

    def __setitem__(self, key: str, file_or_dir):
        if file_or_dir is None:
            pass
        elif isinstance(file_or_dir, Directory):
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
        print(dump(rydanalysis.IO.io.tree(self, include_files)))

    def rmtree(self):
        rmtree(self.path)

    @GetterWithItem
    def iloc(self, index):
        key = list(self.keys())[index]
        return self[key]
