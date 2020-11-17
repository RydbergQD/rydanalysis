import numpy as np
from scipy import ndimage
from scipy import linalg
from rydanalysis import *
from sklearn import decomposition
import xarray as xr


def nn_replace_invalid(data: np.ndarray, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    if invalid is None:
        inv = ~np.isfinite(data)
    else:
        inv = data == invalid

    ind = ndimage.distance_transform_edt(
        inv, return_distances=False, return_indices=True
    )
    return data[tuple(ind)]
