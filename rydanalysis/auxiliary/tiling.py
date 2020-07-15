import functools
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm


def tile_apply(function=None, wx=200, wy=200, stepx=100, stepy=200, col_name='col', row_name='row', tile_name='tile', stack=True):
    """
    Split array into tiles and evaluate function on each tile. Return 2d xr.DataArray of results
    with dim labels ('col', 'row)

    :param function:
    :param wx:
    :param wy:
    :param stepx:
    :param stepy:
    :param func_kwargs:
    :return:
    """

    def _decorate(function):
        @functools.wraps(function)
        def _wrapper(a, **kwargs):
            lx = a.shape[-2]
            ly = a.shape[-1]
            nx_range = np.floor_divide(lx - wx, stepx) + 1
            ny_range = np.floor_divide(ly - wy, stepy) + 1
            out = [[function(a[..., nx * stepx:nx * stepx + wx, ny * stepy:ny * stepy + wy], **kwargs)
                    for nx in range(0, nx_range)] for ny in tqdm(range(0, ny_range))]

            out = xr.concat([xr.concat(row, dim=row_name) for row in out], dim=col_name)
            if not stack:
                return out
            return out.stack({tile_name: (col_name, row_name)})
        return _wrapper
    return _decorate


@xr.register_dataarray_accessor('unique')
class GetUniqueValues:
    def __init__(self, da):
        self.da = da
        pass

    def __call__(self):
        return np.unique(self.da)


@xr.register_dataarray_accessor('iterate')
class IterateOverDim:
    def __init__(self, da):
        self.da = da
        pass

    def __call__(self, dim):
        return self.da.transpose(dim, ...)
