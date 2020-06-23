import xarray as xr
import pandas as pd


from rydanalysis.IO.io import GetterWithTimestamp


@xr.register_dataarray_accessor("ryd_data")
class RydbergDataAccessor:
    TIME_FORMAT = '%Y_%m_%d_%H.%M.%S'

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_netcdf(self, path):
        data = self._obj.reset_index('shot')
        data.to_netcdf(path / 'raw_data.h5')

    @property
    def tmstps(self):
        return pd.DatetimeIndex(self.raw_data.tmstp.values)

    @property
    def parameters(self):
        return self._obj.attrs

    @property
    def variables(self):
        raw_data = self._obj
        coords = raw_data.coords
        dims = coords.dims
        variables = {variable: coords[variable] for variable in coords if variable not in dims}
        return xr.Dataset(variables)

    @GetterWithTimestamp
    def get_single_shot(self, tmstp):
        key = tmstp.strftime(self.TIME_FORMAT)
        path = self.path / 'raw_data' / (key + '.h5')
        with xr.open_dataset(path) as raw_data:
            return raw_data
