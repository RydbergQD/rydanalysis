import xarray as xr
from lmfit import Parameters
from xarray.core.extensions import _register_accessor

Parameters._accessors = dict()


def parameters_to_dataset(params, par_prefix="", par_suffix="", dim_name='fit'):
    if type(params) is not Parameters:
        params = params.params
    return xr.Dataset({
        par_prefix + p.name + par_suffix: (dim_name, [p.value, p.min, p.max, p.init_value, p.stderr, p.vary])
        for p in params.values()},
        coords={dim_name: (dim_name, ['value', 'min', 'max', 'init_value', 'stderr', 'vary'])})


def dataset_to_parameters(ds):
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

def fit_statistics_to_dataset(fit_result):
    ds = xr.Dataset({
        'redchi' : fit_result.redchi,
        'chisqr' : fit_result.chisqr,
        'ressum' : fit_result.residual.sum(),
    })
    return ds

def register_parameters_accessor(name):
    """
     Register a custom accessor on lmfit.Parameters objects.
    :param name: Name under which the accessor should be
    registered. A warning is issued if this name conflicts with a preexisting attribute.
    :return:
    """
    return _register_accessor(name, Parameters)


@xr.register_dataset_accessor('to_parameters')
class DatasetToParameters:
    def __init__(self, ds):
        self.ds = ds
        pass

    def __call__(self):
        return dataset_to_parameters(self.ds)


@register_parameters_accessor('to_dataset')
class ParametersToDataset:
    def __init__(self, params):
        self.params = params
        pass

    def __call__(self):
        return parameters_to_dataset(self.params)


def fit_dataarray(image, model, params):
    fit_result = model.fit(image, params)
    return parameters_to_dataset(fit_result.params)
