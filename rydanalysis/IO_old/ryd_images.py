import xarray as xr
import numpy as np

from rydanalysis.data_structure.ryd_data import RydData


@xr.register_dataset_accessor('ryd_images')
class RydImages(RydData):
    ATOMS_IMAGE = 'image_01'
    LIGHT_IMAGE = 'image_03'
    BACKGROUND_IMAGE = 'image_05'

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self.background = xarray_obj[self.BACKGROUND_IMAGE]
        self.light = xarray_obj[self.LIGHT_IMAGE] - self.background
        self.atoms = xarray_obj[self.ATOMS_IMAGE] - self.background

    @property
    def transmission(self):
        return self.atoms / self.light

    @property
    def optical_depth(self):
        return -np.log(self.transmission)

    @property
    def density(self):
        return (1 + self.saturation_parameter) / self.cross_section * self.optical_depth
