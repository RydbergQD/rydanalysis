import xarray as xr
from lmfit import Parameters
from lmfit.model import ModelResult
from lmfit.printfuncs import gformat
import pandas as pd
from xarray.core.extensions import _register_accessor
from typing import Dict

Parameters._accessors = dict()
parameters_coord_names = ["value", "min", "max", "init_value", "stderr", "vary"]


def parameters_to_dataset(params, par_prefix="", par_suffix="", dim_name="fit"):
    if type(params) is not Parameters:
        params = params.params
    return xr.Dataset(
        {
            par_prefix
            + p.name
            + par_suffix: (
                dim_name,
                [p.value, p.min, p.max, p.init_value, p.stderr, p.vary],
            )
            for p in params.values()
        },
        coords={dim_name: (dim_name, parameters_coord_names)},
    )


def dataset_to_parameters(ds: xr.Dataset, par_prefix="", par_suffix="", dim_name="fit"):
    """
    xr.Dataset needs to be 1d, missing parameter specifications will be silently skipped

    Args:
        ds: dataset that is transformed to parameters
        dim_name: name of coordinates
        par_suffix: Suffix of parameter
        par_prefix: Prefix of parameter
    """
    params = Parameters()
    var: str
    for var in ds:
        if not (var.startswith(par_prefix) and var.endswith(par_suffix)):
            continue
        par_dict = dict()
        for key in ["value", "vary", "min", "max", "expr", "brute_step"]:
            try:
                par_dict[key] = float(ds[var].sel({dim_name: key}))
            except KeyError:
                pass

        var = var[len(par_prefix) : len(var) - len(par_suffix)]
        params.add(var, **par_dict)
    return params


def fit_statistics_to_dataset(fit_result):
    ds = xr.Dataset(
        {
            "redchi": fit_result.redchi,
            "chisqr": fit_result.chisqr,
            "ressum": fit_result.residual.sum(),
        }
    )
    return ds


def register_parameters_accessor(name):
    """
     Register a custom accessor on lmfit.Parameters objects.
    :param name: Name under which the accessor should be
    registered. A warning is issued if this name conflicts with a preexisting attribute.
    :return:
    """
    return _register_accessor(name, Parameters)


@xr.register_dataset_accessor("to_parameters")
class DatasetToParameters:
    def __init__(self, ds):
        self.ds = ds
        pass

    def __call__(self, par_prefix="", par_suffix="", dim_name="fit"):
        return dataset_to_parameters(
            self.ds, par_prefix=par_prefix, par_suffix=par_suffix, dim_name=dim_name
        )


@register_parameters_accessor("to_dataset")
class ParametersToDataset:
    def __init__(self, params):
        self.params = params
        pass

    def __call__(self, par_prefix="", par_suffix="", dim_name="fit"):
        return parameters_to_dataset(
            self.params, par_prefix=par_prefix, par_suffix=par_suffix, dim_name=dim_name
        )


def fit_dataarray(image, model, params):
    fit_result = model.fit(image, params)
    return parameters_to_dataset(fit_result.params)


@register_parameters_accessor("to_report_table")
class ParametersToReportTable:
    def __init__(self, params):
        self.params = params
        pass

    def __call__(self):
        return parameters_to_report_table(self.params)


def parameters_to_report_table(params: Parameters):
    has_err = any([p.stderr is not None for p in params.values()])
    has_expr = any([p.expr is not None for p in params.values()])
    has_brute = any([p.brute_step is not None for p in params.values()])

    headers = ["value"]
    if has_err:
        headers.extend(["standard error", "relative error"])
    headers.extend(["initial value", "min", "max", "vary"])
    if has_expr:
        headers.append("expression")
    if has_brute:
        headers.append("brute step")

    params_table = pd.DataFrame(columns=headers)
    for name, par in params.items():
        params_table.loc[name, "value"] = gformat(par.value)
        if has_err:
            if par.stderr is not None:
                params_table.loc[name, "standard error"] = gformat(par.stderr)
                try:
                    spercent = "({:.2%})".format(abs(par.stderr / par.value))
                except ZeroDivisionError:
                    spercent = ""
                params_table.loc[name, "relative error"] = spercent
        params_table.loc[name, "initial value"] = gformat(par.init_value)
        params_table.loc[name, "min"] = gformat(par.min)
        params_table.loc[name, "max"] = gformat(par.max)
        params_table.loc[name, "vary"] = "%s" % par.vary
        if has_expr:
            expr = ""
            if par.expr is not None:
                expr = par.expr
            params_table.loc[name, "expression"] = expr

        if has_brute:
            brute_step = "None"
            if par.brute_step is not None:
                brute_step = gformat(par.brute_step)
            params_table.loc[name, "brute step"] = brute_step
    return params_table


def merge_fits(fits: Dict[str, ModelResult]):
    fit_list = [fit.params.to_dataset(par_prefix=name) for name, fit in fits.items()]
    return xr.merge(fit_list)
