import xarray as xr
import numpy as np
from typing import Union
from rydanalysis.auxiliary.decorators import cached_property


def load_ryd_data(path):
    data = xr.load_dataset(path)
    coord_keys = [coord for coord in data.coords.keys() if coord not in {'x', 'y', 'time'}]
    data = data.set_index(shot=coord_keys)
    return data


@xr.register_dataset_accessor("ryd_data")
@xr.register_dataarray_accessor("ryd_data")
class RydData:
    """Functionality for xarray datasets including the 'shot' mutiindex."""
    TIME_FORMAT = '%Y_%m_%d_%H.%M.%S'
    SHOT_INDEX = 'shot'
    TMSTP = 'tmstp'

    def __init__(self, xarray_obj):
        self._obj: Union[xr.Dataset, xr.Dataset] = xarray_obj

    @property
    def has_multi_index(self):
        if self.SHOT_INDEX in self._obj.dims:
            return True
        elif self.TMSTP in self._obj.dims:
            return False
        else:
            raise AttributeError("Data has neither the index '{}' nor '{}'".format(
                self.SHOT_INDEX, self.TMSTP
            ))

    @property
    def shot_or_tmstp(self):
        if self.has_multi_index:
            return self.SHOT_INDEX
        else:
            return self.TMSTP

    @property
    def index(self):
        return self._obj[self.shot_or_tmstp].to_index()

    @property
    def variables(self):
        return self.index.to_frame(index=False).set_index(self.TMSTP)

    def to_netcdf(self, path):
        data = self._obj.reset_index(self.SHOT_INDEX)
        data.to_netcdf(path)

    @cached_property
    def image_names(self):
        return [name for name in self._obj.data_vars.keys() if 'image' in name]

    @property
    def has_images(self):
        if len(self.image_names) == 0:
            return False
        else:
            return True

    @property
    def images(self):
        if self.has_images:
            return self._obj[self.image_names]

    @property
    def has_traces(self):
        if 'scope_traces' in self._obj:
            return True
        else:
            return False

    @property
    def date(self):
        tmstp = self._obj.tmstp.values[0]
        time_string = np.datetime_as_string([tmstp], unit='D')
        return time_string[0].replace('-', '_')

    def choose_shot(self, tmstp):
        shot = self._obj.sel(tmstp=tmstp)
        shot.attrs.update(tmstp=tmstp)
        return shot.squeeze(drop=True)


@xr.register_dataarray_accessor("ryd_traces")
class RydTraces(RydData):

    def to_pandas(self, path):
        data = self._obj.reset_index(self.SHOT_INDEX)
        data.to_netcdf(path)
