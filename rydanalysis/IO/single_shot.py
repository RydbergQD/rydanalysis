from rydanalysis.IO.directory import Directory

import pandas as pd
from astropy.io import fits
from os.path import basename


class SingleShot(Directory):
    """
    Analysis of a single experimental run.

    path: folder location of the run.

    Folder has the following structure:
        yyyy_mm_dd_HH.MM.SS:
        - analysis:
          - old_la.csv
          ...
        - exp_data:
          - exp_seq.xml
          - image.fits
          - scope_trace.csv
          - variable.txt
          - voltage.xml
          ...

    """
    def __init__(self, path):
        if not is_single_shot(path):
            raise KeyError("The directory name of a single shot should be in format '%Y_%m_%d_%H.%M.%S'")
        super(SingleShot, self).__init__(path)

    @property
    def old_la(self):
        return pd.read_csv(self['analysis']['old_la.csv'].path, index_col=0, squeeze=True)

    @property
    def image(self):
        with fits.open(self['exp_data']['image.fits'].path) as image:
            image_data = image[0].data
        return image_data

    @property
    def scope_trace(self):
        return pd.read_csv(self['exp_data']['scope_trace.csv'].path, squeeze=True, index_col=0)

    @property
    def parameters(self):
        return pd.read_csv(self['exp_data']['parameters.csv'].path, index_col=0, squeeze=True)

    @property
    def tmstp(self):
        return pd.to_datetime(self.__name__, format='%Y_%m_%d_%H.%M.%S')

    def __repr__(self):
        return "Single shot: " + self.path


def is_single_shot(path):
    name = basename(path)
    try:
        pd.to_datetime(name, format='%Y_%m_%d_%H.%M.%S')
        return True
    except ValueError:
        return False
