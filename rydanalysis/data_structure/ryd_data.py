import xarray as xr
import pandas as pd
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
    def variables(self):
        coords = self._obj.shot.reset_index('shot').coords
        variables = {variable: coords[variable] for variable in coords if variable is not 'shot'}
        df = pd.DataFrame(variables)
        df.set_index('tmstp', inplace=True)
        return df

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


@xr.register_dataarray_accessor("ryd_traces")
class RydTraces(RydData):

    def to_pandas(self, path):
        data = self._obj.reset_index(self.SHOT_INDEX)
        data.to_netcdf(path)

