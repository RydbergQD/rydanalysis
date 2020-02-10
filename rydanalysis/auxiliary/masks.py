import xarray as xr
from abc import ABCMeta, abstractmethod


class Mask(metaclass=ABCMeta):
    def __init__(self, image):
        self.image = image

    @abstractmethod
    def mask(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.image.where(self.mask(*args, **kwargs))


@xr.register_dataset_accessor('rectangular_mask')
@xr.register_dataarray_accessor('rectangular_mask')
class RectangularMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def mask(self, center_x=0, center_y=0, width_x=50, width_y=50):
        image = self.image
        mask = (abs(image.x - center_x) < width_x) * (abs(image.y - center_y) < width_y)
        return mask


@xr.register_dataset_accessor('elliptical_mask')
@xr.register_dataarray_accessor('elliptical_mask')
class EllipticalMask(Mask):
    def __init__(self, image):
        super().__init__(image)

    def mask(self, center_x=0, center_y=0, width_x=50, width_y=50):
        image = self.image
        mask = (((image.x - center_x)/width_x)**2 + ((image.y - center_y)/width_y)**2) < 1
        return mask
