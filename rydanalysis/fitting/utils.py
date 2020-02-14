from xarray import Dataset
from lmfit import Parameters


def params_to_dataset(params, par_prefix="", par_suffix="", dim_name='fit'):
    return Dataset({
        par_prefix + p.name + par_suffix: (dim_name, [p.value, p.min, p.max, p.init_value, p.stderr, p.vary])
        for p in params.values()},
        coords={dim_name: (dim_name, ['value', 'min', 'max', 'init_value', 'stderr', 'vary'])})


def dataset_to_params(ds):
    """
    xr.Dataset needs to be 1d, missing parameter specifications will be silently skipped
    :param ds:
    :return:
    """
    params = Parameters()
    for var in ds:
        par_dict = dict()
        for key in ['value', 'vary', 'min', 'max', 'expr', 'brute_step']:
            try:
                par_dict[key] = float(ds[var].sel(fit=key))
            except KeyError:
                pass
        params.add(var, **par_dict)
    return params
