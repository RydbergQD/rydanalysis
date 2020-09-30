from dataclasses import dataclass
import numpy as np
import xarray as xr
from typing import List
from abc import ABCMeta, abstractmethod
from lmfit.model import Parameters, Model
from lmfit.models import GaussianModel

from rydanalysis.data_analysis.atom_analysis import AbsorptionImaging
from rydanalysis.fitting.fitting_2d import Gaussian2D
from .image_analysis import ImageParameters, analyse_images


class LiveAnalysisDataset(metaclass=ABCMeta):
    def __init__(self, data=None):
        self.ds = None
        if data:
            self.update(data)

    @abstractmethod
    def _analyse(self, data):
        pass

    def update(self, data: xr.Dataset):
        new = self._analyse(data)
        if self.ds:
            self.ds = xr.concat([self.ds, new], dim='shot')
        else:
            self.ds = new


class LiveAnalysisFit(LiveAnalysisDataset):
    def __init__(self, name, model: Model, data=None):
        self.model = model
        self.name = name
        super().__init__(data)

    @property
    def par_prefix(self):
        if self.name:
            return self.name + '_'
        else:
            return ''

    def make_params(self, *args, **kwargs) -> Parameters:
        return self.model.make_params()

    def fit(self, shot):
        params = self.make_params(shot)
        fit = self.model.fit(shot, params)
        return fit.params.to_dataset(par_prefic=self.par_prefix)

    def _analyse(self, data):
        return data.groupby('shot').apply(self.fit)


class LiveAnalysisImages(LiveAnalysisDataset):
    def __init__(self, data=None, parameters=ImageParameters()):
        self.parameters = parameters
        super().__init__(data)

    def make_params(self, shot):
        pass

    def _analyse(self, data):
        fit_ds = analyse_images(shot, self.parameters)
