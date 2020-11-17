import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


@xr.register_dataarray_accessor("gaussian_filter")
class GaussianFilter:
    def __init__(self, image):
        self.image = image
        self.pixel_sizes = np.array([self.pixel_size(dim) for dim in image.dims])

    def __call__(
        self, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0,
    ):
        sigma_in_pixels = sigma / self.pixel_sizes
        result = gaussian_filter(self.image, sigma_in_pixels)
        return self.to_array(result)

    def pixel_size(self, dim):
        image = self.image
        return float((image[dim].max() - image[dim].min()) / (image[dim].size - 1))

    def to_array(self, image):
        dims = self.image.dims
        coords = self.image.coords
        return xr.DataArray(image, dims=dims, coords=coords)
