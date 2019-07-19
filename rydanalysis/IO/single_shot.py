from rydanalysis.IO.directory import Directory

import pandas as pd
from astropy.io import fits
from os.path import basename, join
import numpy as np

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
        image_data = read_fits(self['exp_data']['image.fits'].path)
        return image_data.astype(np.int16)

    @property
    def scope_trace(self):
        return pd.read_csv(self['exp_data']['scope_trace.csv'].path, squeeze=True, index_col=0)

    @property
    def parameters(self):
        return pd.read_csv(self['exp_data']['parameters.csv'].path, index_col=0, squeeze=True)

    @property
    def tmstp(self):
        return pd.to_datetime(self.__name__, format='%Y_%m_%d_%H.%M.%S')

    @property
    def optical_density(self):
        return read_fits(self['analysis']['od.fits'].path) 

    @optical_density.setter
    def optical_density(self, image):
        write_fits(image, join(self['analysis'].path, 'od.fits'))

    def __repr__(self):
        return "Single shot: " + self.path


def is_single_shot(path):
    name = basename(path)
    try:
        pd.to_datetime(name, format='%Y_%m_%d_%H.%M.%S')
        return True
    except ValueError:
        return False


def write_fits(images, path):
    if images.ndim == 2:
        images = [images]
    elif images.ndim == 3:
        pass
    else:
        raise IOError("input must be 2Darray or list of 2Darrays")
    hdul = fits.HDUList([fits.PrimaryHDU(images)])
    hdul.writeto(path,overwrite=True)

def read_fits(path):
    return fits.getdata(path)
# def read_fits(path):
#     with fits.open(path) as image:
#         image_data = image[0].data
#     return image_data
