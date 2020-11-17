import xarray as xr
import pandas as pd


@xr.register_dataarray_accessor("ryd_shot")
class RydbergShotAccessor:
    """"""

    TIME_FORMAT = "%Y_%m_%d_%H.%M.%S"

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
