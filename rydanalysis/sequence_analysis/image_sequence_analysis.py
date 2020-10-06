from dataclasses import dataclass
from typing import List
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from lmfit.models import GaussianModel
from rydanalysis.fitting.utils import merge_fits
from rydanalysis.fitting.fitting_2d.models2d import Gaussian2D
from rydanalysis.data_analysis.dipole_transition import LiveAnalysisTransition
from rydanalysis.data_analysis.atom_analysis import AbsorptionImaging
import numpy as np
from tqdm.notebook import tqdm


@dataclass()
class ImageParameters:
    light_name_index: int = 3
    atom_name_index: int = 1
    background_name_index: int = 5
    mask: List = None
    slice_option: str = "maximum"
    slice_position_x: float = None
    slice_position_y: float = None

    SLICE_OPTIONS = pd.Series(["maximum", "central moment", "manual"])

    @property
    def light_name(self) -> str:
        index = self.light_name_index
        return self.index_to_name(index)

    @property
    def atom_name(self) -> str:
        index = self.atom_name_index
        return self.index_to_name(index)

    @property
    def background_name(self) -> str:
        index = self.background_name_index
        return self.index_to_name(index)

    @staticmethod
    def index_to_name(index: int) -> str:
        return "image_" + str(index).zfill(2)

    @staticmethod
    def name_to_index(name: str) -> int:
        return int(name[-2:])

    def analyse_images(self, images, t_exp):
        imaging = self.get_absorption(images, t_exp)

        density = imaging.density
        density_masked = self._apply_mask(density)

        res = [self._apply_fits(image) for tmstp, image
               in tqdm(density_masked.groupby("tmstp"), desc='fit images', leave=False)]
        fit_ds, summary = list(zip(*res))
        fit_ds = xr.concat(fit_ds, dim="tmstp")
        summary = xr.DataArray(pd.concat(summary, axis=1), dims=["variable", "tmstp"])
        return fit_ds, summary

    def get_absorption(self, images, t_exp):

        binning = 100 / images.x.size
        background = images[self.background_name]

        absorption_images = images[self.atom_name] - background
        light_images = images[self.light_name] - background

        return AbsorptionImaging(absorption_images, light_images, t_exp=t_exp, binning=binning,
                                 transition=LiveAnalysisTransition())

    def _apply_mask(self, dataset_or_array):
        mask_vertices = self.mask
        masked = dataset_or_array.polygon_mask.apply_mask(mask_vertices)
        return masked.dropna(dim='x', how='all').dropna(dim='y', how='all')

    def _find_center(self, image: xr.DataArray):
        slice_option = self.slice_option
        mask_vertices = self.mask

        if slice_option == "maximum":
            return find_maximum(image, mask_vertices)
        elif slice_option == "manual":
            position_x = self.slice_position_x
            position_y = self.slice_position_y
            c = image.sel(x=position_x, y=position_y, method='nearest')
            return xr.Dataset(dict(x=c.x.values, y=c.y.values, amp=c.values),
                              coords={"tmstp": image.tmstp})
        else:
            raise NotImplementedError("This method is not yet implemented :(")

    def _apply_fits(self, image):
        center = self._find_center(image)
        fit_x = fit_slice(image, center, 'x')
        fit_y = fit_slice(image, center, 'y')
        fit_2d = fit_2d_gaussian(image, fit_x, fit_y)

        fits = {"2d_": fit_2d, "slice_x_": fit_x, "slice_y_": fit_y}
        fit_ds = merge_fits(fits)
        fit_ds = fit_ds.assign_coords(dict(tmstp=image.tmstp))
        summary = summarize_fit(fit_ds, float(image.tmstp), center)
        return fit_ds, summary


def find_maximum(image, mask_vertices):
    filtered = gaussian_filter(image, sigma=2, mode='nearest')
    filtered = xr.DataArray(filtered, coords=image.coords)
    image_masked = image.polygon_mask.apply_mask(mask_vertices)

    maxima = peak_local_max(filtered.values)
    maximum = max([image_masked[tuple(m)] for m in maxima])
    return maximum


def fit_slice(image: xr.DataArray, center, variable='x'):
    fixed = 'y' if variable == 'x' else 'x'
    slice_x = image.sel({fixed: center[fixed].values}, drop=True, method='nearest')
    slice_x = slice_x.dropna(dim=variable)
    model = GaussianModel()

    params = model.guess(gaussian_filter(slice_x.values, sigma=1),
                         x=slice_x[variable].values)
    return model.fit(slice_x.values, x=slice_x[variable], params=params, nan_policy='omit')


def fit_2d_gaussian(image, fit_x, fit_y):
    model = Gaussian2D()
    amp = np.mean([fit_x.best_values['amplitude'], fit_y.best_values['amplitude']])
    cen_x = fit_x.best_values['center']
    cen_y = fit_y.best_values['center']
    sig_x = fit_x.best_values['sigma']
    sig_y = fit_y.best_values['sigma']

    params = model.make_params(amp=amp, cen_x=cen_x, cen_y=cen_y, sig_x=sig_x, sig_y=sig_y)
    return model.fit(image, params=params)


def summarize_fit(fit_ds, tmstp, center):
    summary = fit_ds.to_array().sel(fit='value', drop=True).to_series()
    summary["center_x"] = center.x.values
    summary["center_y"] = center.y.values
    summary.name = tmstp
    return summary
