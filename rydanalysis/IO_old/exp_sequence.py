from os.path import join, basename

import pandas as pd
import xarray as xr

from rydanalysis.IO_old.os import Directory
from rydanalysis.IO_old.io import GetterWithTimestamp


class ExpSequence(Directory):
    TIME_FORMAT = "%Y_%m_%d_%H.%M.%S"

    def __init__(self, path, lazy=False):
        super(ExpSequence, self).__init__(path)
        # create sequence analysis dir
        # self.analysis = self['analysis']
        self.raw_data = self._open_raw_data()
        if not lazy:
            self.raw_data.load()
        # self.averaged_images = self.analysis['averaged_images']

    def _open_raw_data(self):
        if (self.path / "raw_data.h5").is_file():
            ds = xr.open_dataset(self.path / "raw_data.h5")
            ds = self._create_multiindex(ds)
            return ds
        path = self.path / "raw_data"
        ds = xr.open_mfdataset(
            path.glob("*.h5"), concat_dim="tmstp", combine="nested", parallel=True
        )
        ds = self._create_multiindex(ds)
        return ds

    @staticmethod
    def _create_multiindex(ds):
        coord_keys = [
            coord for coord in ds.coords.keys() if coord not in {"x", "y", "time"}
        ]
        # coord_keys.remove('x')
        # coord_keys.remove('y')
        # coord_keys.remove('time')
        ds = ds.set_index(shot=coord_keys)
        return ds

    @property
    def tmstps(self):
        return pd.DatetimeIndex(self.raw_data.tmstp.values)

    @property
    def parameters(self):
        return self.raw_data.attrs

    @property
    def variables(self):
        raw_data = self.raw_data
        coords = raw_data.coords
        dims = coords.dims
        variables = {
            variable: coords[variable] for variable in coords if variable not in dims
        }
        return xr.Dataset(variables)

    def __repr__(self):
        return "Experimental Sequence: " + str(self.path)

    @GetterWithTimestamp
    def get_single_shot(self, tmstp):
        key = tmstp.strftime(self.TIME_FORMAT)
        path = self.path / "raw_data" / (key + ".h5")
        with xr.open_dataset(path) as raw_data:
            return raw_data


def is_exp_sequence(path):
    directory = Directory(path)
    for path in directory.iter_dirs():
        if is_single_shot(path):
            return True
    return False


def is_single_shot(path):
    name = basename(path)
    try:
        pd.to_datetime(name, format="%Y_%m_%d_%H.%M.%S.h5")
        return True
    except ValueError:
        return False
