import numpy as np
from scipy import ndimage
from scipy import linalg
from rydanalysis import *
from sklearn import decomposition

def absorbtion_to_OD(image):
    return -np.log(1 - image)


def transmission_to_OD(image):
    return -np.log(image)


def crop_image(image, xslice=slice(0, -1), yslice=slice(0, -1)):
    return image[(xslice, yslice)]


def calc_transmission(im):
    bg = im['image_05'].values
    light = im['image_03'].values - bg
    atoms = im['image_01'].values - bg
    trans = atoms / light
    # np.place(trans,trans>=1,1)
    # np.place(trans,light==0,1)
    # np.place(trans,trans<=0,0.0001)
    return trans


def elliptical_mask(shape, x0=0, y0=0, a=1, b=1):
    nx, ny = shape
    x, y = np.mgrid[-x0:nx - x0, -y0:ny - y0]
    mask = x * x / a ** 2 + y * y / b ** 2 >= 1
    return mask


def agnostic_select(data, selector):
    if type(selector) == slice:
        selection = data[selector]
    elif type(selector) == dict:
        selection = data.sel(**selector)
    elif type(selector) == int:
        selection = data[0:selector]
    elif type(selector) == tuple and len(selector) == 2:
        selection = data[selector[0]:selector[1]]
    else:
        raise TypeError('selector has to be a slice, int or dict')
    return selection


def nn_replace_invalid(data, invalid=None):
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

    ind = ndimage.distance_transform_edt(inv,
                                         return_distances=False,
                                         return_indices=True)
    return data[tuple(ind)]


def prepare_ref_basis(ref_images, mask=None):
    """deprecated"""
    ref_images = ref_images.astype(float)
    n, x, y = ref_images.shape
    if mask is None:
        mask = np.full((x, y), True)
    R = ref_images.reshape(n, x * y)
    Rm = R[:, np.ravel(mask)]
    B = Rm @ Rm.T
    B_inv = linalg.pinv(B)
    return B_inv, R


def calc_ref_image(image, B_inv=None, R=None, mask=None, coefficients=False):
    """deprecated"""
    if mask is None:
        mask = ~np.zeros_like(image, dtype=bool)
    k = np.ravel(mask)
    Rm = R[:, k]
    A = image.ravel()
    Am = A[k]
    c = B_inv @ (Rm @ Am)
    R_opt = (c @ R)
    R_opt = R_opt.reshape(image.shape)
    if coefficients:
        return R_opt, c
    else:
        return R_opt


def ref_images_truncated_svd(a, b, mask=None, n_components=None):
    """
    Fit b[,:mask] with the first #n_basis principal components of a. Uses Truncated SVA
    for the decomposition and linear least squares for the fit.
    Args:
        a: 3d array with shape (n_a_samples, n_x, n_y)
        b: 3d array with shape (n_b_samples, n_x, n_y)
        mask: 2d array with shape (n_x,n_y)
        n_components (int): number of components to use

    Returns:
        fit to b (not masked), 3d array with the same shape as b

    """

    shape = a[0].shape
    if mask is None:
        mask = np.full(shape, True)

    # flatten 2d samples
    a_flat = np.reshape(a, (a.shape[0], -1))

    # singular value decomposition
    decomp = decomposition.TruncatedSVD(n_components=n_components)
    decomp = decomp.fit(a_flat)

    # linear least squares fit
    coeff = np.linalg.lstsq(decomp.components_[:, np.ravel(mask)].T, b[:, mask].T, rcond=None)[0]
    fit = coeff.T @ decomp.components_

    # reshaping to list of 2d samples
    fit = fit.reshape((-1, *shape))

    return fit
