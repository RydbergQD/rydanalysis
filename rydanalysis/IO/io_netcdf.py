from rydanalysis.IO.os import Directory

import xarray as xr
import pandas as pd
import numpy as np

class NetCDFDir(Directory):
    def __init__(self, path):
        super(NetCDFDir,self).__init__(path)
        
    def get_attrs(self,ds):
        return [k for k in ds.coords if len(np.unique(ds[k].values))==1 ]
    
    @property
    def data(self):
        ds = xr.concat(self.values(),dim='timestamp')
        attrs = self.get_attrs(ds)
        ds = ds.drop(attrs)
        return ds
