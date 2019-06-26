from rydanalysis.IO.directory import Directory
from rydanalysis.IO.single_shot import SingleShot

import os
from os.path import join


class SingleShotWithItem:
    def __init__(self, get_path):
        self.get_path = get_path

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        path = self.get_path(instance)
        return SingleShot(path)


class ExpSequence(Directory):
    def __init__(self, path):
        super(ExpSequence, self).__init__(path)

    def __getitem__(self, key):
        path = join(self.path, key)
        if os.path.isdir(path):
            return SingleShot(path)
        else:
            raise KeyError(key)

    def __setitem__(self, key: str, single_shot):
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
            yield folder

    def iter_dirs(self):
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

    def structure(self, include_files='all'):
        print(dump(dir_to_dict(self.path, include_files)))

    def rmtree(self):
        rmtree(self.path)

    @GetterWithItem
    def iloc(self, index):
        key = list(self.keys())[index]
        return self[key]