from typing import List, Iterable
import numpy as np

import pandas as pd
from pathlib import Path
import xarray as xr
import dask
from .read_images import read_images
from .read_traces import read_traces

# Get tmstps


def read_tmstps_txt(
    _path,
    filename_pattern: str = "????_??_??_??.??.??",
    strftime: str = "%Y_%m_%d_%H.%M.%S",
):
    path = _path / "Variables"
    tmstps = []
    for sub_path in path.glob(filename_pattern + ".txt"):
        try:
            tmstps.append(pd.to_datetime(sub_path.name, format=strftime + ".txt"))
        except ValueError:
            print("couldn't read {0}. Skipping this file...".format(sub_path.name))
    return tmstps


def analyze_existing_h5(destiny_path: Path) -> Iterable[pd.DatetimeIndex]:
    try:
        data = load_data(destiny_path, lazy=True, to_multiindex=False)
        time = update_time(data)
    except OSError:
        return [], None
    tmstps = map(pd.to_datetime, data.tmstp.values)
    return tmstps, time


def update_time(data):
    try:
        return data.time
    except KeyError:
        return None


def compare_tmstps(new_tmstps, old_tmstps):
    new_tmstps = set(new_tmstps)
    tmstp = list(new_tmstps.difference(old_tmstps))
    tmstp.sort()
    return tmstp


# Get parameters


def read_parameters_single(
    tmstp: pd.Timestamp, path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S", **csv_kwargs
) -> pd.Series:
    parameter_path = path / "Variables" / tmstp.strftime(strftime + ".txt")
    parameters = pd.read_csv(parameter_path, **csv_kwargs)
    parameters.name = tmstp
    parameters.index.name = None
    parameters.drop("dummy", inplace=True)
    return parameters


def read_parameters(
    tmstps, path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S", **csv_kwargs
):
    _parameters = [
        read_parameters_single(tmstp, path, strftime, **csv_kwargs) for tmstp in tmstps
    ]
    _parameters = pd.concat(_parameters, axis=1)
    _parameters = _parameters.T
    _parameters.index.name = "tmstp"
    _parameters.columns.name = "param_dim"
    return _parameters


def get_unique_parameters(parameters):
    _parameters = parameters
    parameters = _parameters.iloc[0].loc[_parameters.nunique() == 1]
    return parameters


def get_shot_multiindex(parameters):
    variables = parameters.loc[:, parameters.nunique() > 1]
    return pd.MultiIndex.from_frame(variables.reset_index())


def load_data(path, lazy=None, to_multiindex=True):
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        data = xr.open_mfdataset(
            path.glob("*.h5"), join="left", data_vars="minimal", chunks={"tmstp": 1}, engine="netcdf4"
        )
    if lazy is None:
        lazy = len(data.tmstp) > 500
    if not lazy:
        data = data.load()
    if to_multiindex:
        data = raw_data_to_multiindex(data)
    return data


def raw_data_to_multiindex(data: xr.Dataset):
    parameters = data.parameters.to_pandas()
    data = data.drop_vars("parameters")
    data = data.drop_dims("param_dim")
    get_unique_parameters(parameters)
    multi_index = get_shot_multiindex(parameters)
    new_indices = multi_index.to_frame().set_index("tmstp")
    data = data.merge(new_indices)
    return data.set_index(shot=("tmstp", *new_indices.columns))


# Get raw data


def read_raw_data(
    tmstps: List[pd.Timestamp],
    path: Path,
    strftime: str = "%Y_%m_%d_%H.%M.%S",
    csv_kwargs=None,
    fast_csv_kwargs=None,
    times=None,
    interface: str = "notebook",
):
    raw_data = xr.Dataset()
    images = read_images(tmstps, path, strftime, interface)
    if images is not None:
        raw_data = xr.merge([raw_data, images])
    traces = read_traces(
        tmstps, path, times, strftime, csv_kwargs, fast_csv_kwargs, interface
    )
    if traces is not None:
        raw_data["scope_traces"] = traces
    raw_data["parameters"] = read_parameters(tmstps, path, strftime, **csv_kwargs)
    return raw_data


def update_data(
    path, csv_path="peak_df.csv",
    height=0.0008,
    prominence=0.0001,
    threshold=None,
    distance=None,
    width=None,
    sign=-1,
    freq1=1, freq2=1/2e-3
):
    data = load_data(path, to_multiindex=False, lazy=None)
    try:
        peak_df = pd.read_csv("peak_df.csv")
        print("Found exisiting peak_df.")
        existing_tmstps = set(np.unique(pd.to_datetime(np.unique(peak_df["tmstp"]))))
        all_tmstps = set(data.tmstp.values)
        new_tmstps = list(all_tmstps - existing_tmstps)
        shot = get_shot_multiindex(data.parameters.to_pandas())
        peak_df = peak_df.set_index(["peak_number"] + list(shot.names))
        if len(new_tmstps) != 0:
            print("Add new peaks to df...")
            new_data = data.sel(tmstp=new_tmstps, drop=True)
            new_data = raw_data_to_multiindex(new_data)
            traces = new_data.scope_traces
            traces["time"] = traces.time * 1e6
            new_peak_df = traces.peaks_summary.get_peak_description(
                height, prominence, threshold, distance, width, sign=sign,
                freq1=freq1, freq2=freq2
            )
            peak_df = pd.concat([peak_df, new_peak_df])
            peak_df.to_csv("peak_df.csv")
        data = raw_data_to_multiindex(data)
    except FileNotFoundError:
        data = raw_data_to_multiindex(data)
        traces = data.scope_traces
        traces["time"] = traces.time * 1e6
        peak_df = traces.peaks_summary.get_peak_description(
            height, prominence, threshold, distance, width, sign=sign,
                freq1=freq1, freq2=freq2
        )
        peak_df.to_csv("peak_df.csv")
    return data, peak_df