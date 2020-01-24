import numpy as np
import pandas as pd
import xarray as xr
import copy
import functools


def apply_on_da(func):
    @functools.wraps(func)
    def wrapper(da, *args, **kwargs):
        a = func(da.values, *args, **kwargs)
        da = xr.DataArray(data=a, coords=da.coords, dims=da.dims)
        return da

    return wrapper


def calc_mean_seq(images):
    return np.nanmean(images, axis=0)


def calc_std_seq(images):
    return np.nanstd(images, axis=0)


def remove_fringes(image, base):
    raise NotImplementedError("Fringe removal is not yet implemented")


def stderr_weighted_average(g):
    rel_err = g.amp.stderr / g.amp.value
    weights = 1 / rel_err
    return (g.image_od * weights).sum() / weights.sum()


def fittoSeries(fit):
    p = fit.params
    dict_p = {key: par2dict(p[key]) for key in p}
    flat_dict = flatten_dict(dict_p)
    df = pd.DataFrame(flat_dict, index=[0])
    # ps = pd.Series(flatten_dict(dict_p))
    return df.squeeze()


def fit_to_dataset(fit):
    p = fit.params
    dict_p = {key: par2dict(p[key]) for key in p}
    data = {k: ('quantity', list(v.values())) for k, v in dict_p.items()}
    coords = list(next(iter(dict_p.values())).keys())
    ds = xr.Dataset(data_vars=data, coords={'quantity': coords})
    return ds


def da_apply_fit(da, params, model_class, mask=None, return_type='params'):
    """

    Args:
        mask: if provided, fit only the masked area
        da: 3d xr.DataArray of single_image with shape (n_shots, n_pixel_x, n_pixel_y)
        params: fit parameters
        model_class: fitting model to use

    Returns:
        xr.Dataset of the best fit parameters

    """
    p_list = list()
    for image in da:
        arr = copy.deepcopy(image.values)
        if mask is not None:
            np.putmask(arr, ~mask, np.nan)
        fit = model_class(image, params=params)
        params = fit.fit_data().params
        p_list.append(xr.Dataset(params.valuesdict(), coords=image.coords))
    return xr.concat(p_list, dim=da.dims[0])




def iteritems_nested(d):
    def fetch(suffixes, v0):
        if isinstance(v0, dict):
            for k, v in v0.items():
                for i in fetch(suffixes + [k], v):  # "yield from" in python3.3
                    yield i
        else:
            yield (suffixes, v0)

    return fetch([], d)


def flatten_dict(d):
    return {tuple(ks): v for ks, v in iteritems_nested(d)}


def par2dict(p):
    return dict(
        value=p.value,
        min=p.min,
        max=p.max,
        init_value=p.init_value,
        stderr=p.stderr,
        # correl = p.correl,
        vary=p.vary,
    )
