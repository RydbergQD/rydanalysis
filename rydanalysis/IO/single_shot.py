from rydanalysis.IO.directory import Directory

import pandas as pd
from astropy.io import fits


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
    def variable(self):
        return pd.read_csv(self['exp_data']['variables.csv'].path, index_col=0, squeeze=True)

    @property
    def tmstp(self):
        return pd.to_datetime(self.__name__, format='%Y_%m_%d_%H.%M.%S')
