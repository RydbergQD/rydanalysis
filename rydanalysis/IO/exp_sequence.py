from rydanalysis.IO.os import Directory
from rydanalysis.IO.single_shot import SingleShot, is_single_shot, read_fits, write_fits

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
        # create sequence analysis dir
       # self.analysis = self['analysis']
        self.raw_data = self['raw_data']
       # self.averaged_images = self.analysis['averaged_images']

        self.parameters = self.raw_data.parameters
        self.variables = self.raw_data.variables
        #self.var_grid = self.raw_data.var_grid

    def __repr__(self):
        return "Experimental Sequence: " + self.path

    def get_single_shot(self, tmstp):
        key = tmstp.strftime('%Y_%m_%d_%H.%M.%S')
        folder = join(self.path, key)
        if is_single_shot(folder):
            return SingleShot(folder)

    def iter_matching_shots(self, **var):
        """
        iterate over shots that mach var
        """
        selection = match_rows(self.variables, **var)
        for tmstp in selection.index:
            key = tmstp.strftime('%Y_%m_%d_%H.%M.%S')
            folder = join(self.path, key)
            if is_single_shot(folder):
                yield SingleShot(folder)

    def get_averaged_image(self, var):
        num = match_rows(self.var_grid, var).index[0]
        image = read_fits(self.averaged_images['averaged_od_{:04d}.fits'.format(num)].path)
        return image
    
    def set_averaged_image(self, image, var):
        num = match_rows(self.var_grid, var).index[0]
        print(num)
        write_fits(image, join(self.averaged_images.path, 'averaged_od_{:04d}.fits'.format(num)))

    def iter_averaged_images(self):
        for key in os.listdir(self.averaged_images.path):
            path = join(self.averaged_images.path, key)
            yield read_fits(path)


def is_exp_sequence(path):
    directory = Directory(path)
    for path in directory.iter_dirs():
        if is_single_shot(path):
            return True
    return False


def match_rows(df, **var):
    """
    return those rows of a dataframe that match the var (dict)
    """
    select = (df[var].values == list(var.values()))
    select = select.all(axis=1)
    return df[select]
