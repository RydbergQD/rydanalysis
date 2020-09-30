import xarray as xr
from abc import ABCMeta, abstractmethod
import numpy as np
from skimage.draw import polygon2mask


class Mask(metaclass=ABCMeta):
    def __init__(self, image):
        self.image = image

    @abstractmethod
    def get_mask(self, *args, **kwargs):
        pass

    def apply_mask(self, *args, **kwargs):
        return self.image.where(self.get_mask(*args, **kwargs))


@xr.register_dataset_accessor('rectangular_mask')
@xr.register_dataarray_accessor('rectangular_mask')
class RectangularMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, center_x=0, center_y=0, width_x=50, width_y=50):
        image = self.image
        mask = (abs(image.x - center_x) < width_x) * (abs(image.y - center_y) < width_y)
        return mask


@xr.register_dataset_accessor('polygon_mask')
@xr.register_dataarray_accessor('polygon_mask')
class PolygonMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    @staticmethod
    def nearest_index(value, coordinate_array):
        return abs(value - coordinate_array).argmin()

    def get_mask(self, coord_vertices):
        if coord_vertices is None:
            return True
        x = self.image.x.values
        y = self.image.y.values
        shape = (len(x), len(y))
        coord_vertices = np.array(coord_vertices)

        x_coords = [self.nearest_index(coord, x) for coord in coord_vertices[:, 0]]
        y_coords = [self.nearest_index(coord, y) for coord in coord_vertices[:, 1]]
        vertices = np.array([x_coords, y_coords]).T

        polygon = polygon2mask(shape, vertices)
        mask = xr.DataArray(polygon, coords={'x': x, 'y': y}, dims=['x', 'y'])
        return mask


@xr.register_dataset_accessor('elliptical_mask')
@xr.register_dataarray_accessor('elliptical_mask')
class EllipticalMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, center_x=0, center_y=0, width_x=50, width_y=50, x='x', y='y', invert=False):
        image = self.image
        mask = (((image[x] - center_x)/width_x)**2 + ((image[y] - center_y)/width_y)**2) < 1
        if invert:
            return ~mask
        else:
            return mask


@xr.register_dataset_accessor('eit_mask')
@xr.register_dataarray_accessor('eit_mask')
class EITMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def get_mask(self, center_x=0, center_y=0, width_x=200, width_y=600, width_eit=50):
        image = self.image
        mask_cloud = (((image.x - center_x)/width_x)**2 + ((image.y - center_y)/width_y)**2) < 1
        mask_spot = (((image.x - center_x)/width_eit)**2 + ((image.y - center_y)/width_eit)**2) > 1
        return mask_cloud*mask_spot
