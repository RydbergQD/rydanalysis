from rydanalysis.IO.h5 import H5File
from rydanalysis.IO.os import Directory

import pandas as pd
import xarray as xr
from astropy.io import fits
from os.path import basename, dirname, join
import h5py


class SingleShot(H5File):
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
    time_format = '%Y_%m_%d_%H.%M.%S'

    def __init__(self, path):
        super(SingleShot, self).__init__(path)
        with h5py.File(path, 'r') as hf:
            if not hf.attrs['ryd_type'] == 'single_shot':
                raise TypeError("The H5 Dataset is not a single Shot")
            self.tmstp = pd.to_datetime(hf.attrs['tmstp'], format=self.time_format)

    @classmethod
    def initiate_new(cls, path, tmstp: pd.Timestamp):
        tmstp_str = tmstp.strftime(cls.time_format)
        file_path = join(path, tmstp_str + '.h5')
        with h5py.File(file_path, 'w') as hf:
            hf.attrs['ryd_type'] = 'single_shot'
            hf.attrs['tmstp'] = tmstp_str
        instance = cls(file_path)
        return instance

    @property
    def old_la(self):
        return Directory(dirname(dirname(self.path)))['old_la.csv']

    @property
    def images(self):
        return xr.Dataset({key: (['x', 'y'], image) for key, image in self['images'].items()})

    @property
    def scope_trace(self):
        return self['scope_trace']

    @property
    def parameters(self):
        return self['parameters']

    def __repr__(self):
        return "Single shot: " + self.path

    def __str__(self):
        return "Single shot: " + self.__name__


def is_single_shot(path):
    name = basename(path)
    try:
        pd.to_datetime(name, format='%Y_%m_%d_%H.%M.%S.h5')
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
    hdul.writeto(path)


def read_fits(path):
    with fits.open(path) as image:
        image_data = image[0].data
    return image_data
