from rydanalysis.IO.directory import Directory, File
from rydanalysis.IO.single_shot import SingleShot, is_single_shot

import os
from os.path import join, isfile, isdir
import pandas as pd


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
        self.parameters, self.variables = self.parse_variables()

    def parse_variables(self):
        parameters = pd.DataFrame()
        for single_shot in self.iter_single_shots():
            parameter = single_shot.parameters
            parameter.name = single_shot.tmstp
            parameters = pd.concat([parameters, parameter], axis=1, sort=False)

        parameters = parameters.T
        parameters.index.name = 'tmstp'
        parameters.columns.name = None
        # noinspection PyTypeChecker
        variables = parameters.T[parameters.nunique() > 1].T
        try:
            del variables['dummy']
        except KeyError:
            pass
        # noinspection PyTypeChecker
        parameters = parameters.T[parameters.nunique() == 1].T.mean(axis=0)
        return parameters, variables

    def __getitem__(self, key):
        path = join(self.path, key)
        if isfile(path):
            return File(path)
        elif isdir(path):
            if is_single_shot(path):
                return SingleShot(path)
            if is_exp_sequence(path):
                return ExpSequence(path)
            return Directory(path)
        raise KeyError(key)

    def iter_single_shots(self):
        for key in os.listdir(self.path):
            folder = join(self.path, key)
            if is_single_shot(folder):
                yield SingleShot(folder)

    def __repr__(self):
        return "Experimental Sequence: " + self.path

    def iter_matching_shots(self, var):
        """
        iterate over shots that mach var
        """
        selection = self.variables[self.variables[var].values == list(var.values())]
        for tmstp in selection.index:
            key=tmstp.strftime('%Y_%m_%d_%H.%M.%S')
            folder = join(self.path, key)
            print(folder)
            if is_single_shot(folder):
                yield SingleShot(folder)
    

def is_exp_sequence(path):
    directory = Directory(path)
    for path in directory.iter_dirs():
        if is_single_shot(path):
            return True
    return False
