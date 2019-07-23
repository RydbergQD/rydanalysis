from rydanalysis.IO.os import Directory

import xarray as xr
import pandas as pd
import numpy as np


class RawData(Directory):
    def __init__(self, path):
        super(RawData, self).__init__(path)
        self.variables, self.parameters = self.get_parameters()
        self.var_grid = self.get_var_grid()

    def __repr__(self):
        return "Raw data: " + self.path

    def get_parameters(self):
        _parameters = pd.concat([single_shot.parameters for single_shot in self.values()], axis=1)
        variables = _parameters[_parameters.T.nunique() > 1].T
        parameters = _parameters[_parameters.T.nunique() == 1].mean(axis=1)
        return variables, parameters

    def get_var_grid(self):
        # create meshgrid of variable values as df
        list_val = [np.array(list(set(self.variables[col].values))) for col in self.variables]
        val_mesh = np.meshgrid(*list_val)
        var_grid = pd.DataFrame(dict(zip(self.variables, val_mesh)))
        return var_grid

    def get_scope_traces(self):
        return pd.concat([single_shot.scope_trace for single_shot in self.values()], axis=1)

    def get_images(self):
        index = pd.MultiIndex.from_frame(self.variables.reset_index())
        return xr.concat([single_shot.images for single_shot in self.values()], dim=index)
