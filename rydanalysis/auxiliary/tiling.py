import functools
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

def tile_apply(function=None, wx=200, wy=200, stepx=100, stepy=200):
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

            return xr.concat([xr.concat(row, dim='col') for row in out], dim='row')
        return _wrapper
    return _decorate
