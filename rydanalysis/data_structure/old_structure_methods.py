from typing import Tuple, List

import lecroyparser
import pandas as pd
from pathlib import Path
import astropy.io.fits
import numpy as np
import xarray as xr
from rydanalysis.auxiliary.user_input import custom_tqdm


# Get parameters

def extract_tmstps(_path, filename_pattern: str = '????_??_??_??.??.??', strftime: str = '%Y_%m_%d_%H.%M.%S'):
    path = _path / 'Variables'
    tmstps = []
    for sub_path in path.glob(filename_pattern + ".txt"):
        try:
            tmstps.append(pd.to_datetime(sub_path.name, format=strftime + '.txt'))
        except ValueError:
            print("couldn't read {0}. Skipping this file...".format(sub_path.name))
    return tmstps


def read_parameters_single(tmstp: pd.Timestamp, path: Path,
                           strftime: str = '%Y_%m_%d_%H.%M.%S', **csv_kwargs) -> pd.Series:
    parameter_path = path / 'Variables' / tmstp.strftime(strftime + '.txt')
    parameters = pd.read_csv(parameter_path, **csv_kwargs)
    parameters.name = tmstp
    parameters.index.name = None
    parameters.drop('dummy', inplace=True)
    return parameters


def read_parameters(tmstps, path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S', **csv_kwargs):
    _parameters = [read_parameters_single(tmstp, path, strftime, **csv_kwargs) for tmstp in tmstps]
    _parameters = pd.concat(_parameters, axis=1)
    _parameters = _parameters.T
    _parameters.index.name = 'tmstp'
    _parameters.columns.name = "param_dim"
    return _parameters


# Read scope traces


def read_trace_csv(tmstp: pd.Timestamp, path: Path,
                   strftime: str = '%Y_%m_%d_%H.%M.%S', **csv_kwargs):
    traces_path = path / 'Scope Traces'
    file = traces_path / tmstp.strftime(strftime + '_C1.csv')
    if not file.is_file():
        pass
    try:
        return pd.read_csv(file, **csv_kwargs)
    except OSError:
        pass


def read_trace_csv_values(tmstp: pd.Timestamp, path: Path,
                          strftime: str = '%Y_%m_%d_%H.%M.%S', **csv_kwargs):
    scope_trace = read_trace_csv(tmstp, path, strftime, **csv_kwargs)
    if scope_trace is None:
        return None
    if scope_trace.shape[0] > 1:
        scope_trace.name = tmstp
        return scope_trace


def read_traces_index_csv(tmstps, path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S',
                          **csv_kwargs):
    for tmstp in tmstps:
        trace = read_trace_csv(tmstp, path, strftime, **csv_kwargs)
        if trace is not None:
            return trace.index.values


def read_trace_trc(tmstp, path, strftime: str = '%Y_%m_%d_%H.%M.%S', channel="C3"):
    traces_path = path / 'Scope Traces'
    file = traces_path / tmstp.strftime(channel + strftime + '00000.trc')
    try:
        return lecroyparser.ScopeData(str(file))
    except FileNotFoundError:
        return None


def read_scope_trace_trc_values(tmstp, path, strftime: str = '%Y_%m_%d_%H.%M.%S', channel="C3"):
    data = read_trace_trc(tmstp, path, strftime, channel)
    if data is not None:
        return data.y.astype(np.float16)


def read_traces_index_trc(tmstps, path, strftime: str = '%Y_%m_%d_%H.%M.%S', channel="C3"):
    for tmstp in tmstps:
        data = read_trace_trc(tmstp, path, strftime, channel)
        if data is not None:
            return data.x


def find_valid_trace(path):
    traces_path = path / 'Scope Traces'
    for file in traces_path.glob("*.trc"):
        channel = file.name[:2]
        return "trc", channel
    for file in traces_path.glob("*_C1.csv"):
        channel = file.stem[-2:]
        return "csv", channel


def initialize_traces(tmstps, time):
    if time is None:
        return None
    shape = (len(tmstps), time.size)

    scope_traces = xr.DataArray(
        np.full(shape, np.NaN, dtype=np.float32),
        dims=['tmstp', 'time'],
        coords={'tmstp': tmstps, 'time': time},
    )
    return scope_traces


def read_traces(tmstps, path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S', csv_kwargs=None,
                fast_csv_kwargs=None, interface="notebook"):
    if csv_kwargs is None:
        csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    if fast_csv_kwargs is None:
        fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)
    method, channel = find_valid_trace(Path(path))
    if method == "csv":
        return read_traces_csv(tmstps, path, strftime, csv_kwargs, fast_csv_kwargs, interface)
    else:
        return read_traces_trc(tmstps, path, strftime, channel, interface)


def read_traces_csv(tmstps, path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S', csv_kwargs=None,
                    fast_csv_kwargs=None, interface="notebook"):
    if csv_kwargs is None:
        csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    if fast_csv_kwargs is None:
        fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)
    time = read_traces_index_csv(tmstps, path, strftime, **csv_kwargs)
    scope_traces = initialize_traces(tmstps, time)
    if scope_traces is None:
        return None
    for tmstp in custom_tqdm(tmstps, interface, 'load scope traces', leave=True):
        trace = read_trace_csv_values(tmstp, path, strftime, **fast_csv_kwargs)
        scope_traces.loc[{'tmstp': tmstp}] = trace
    return scope_traces


def read_traces_trc(tmstps, path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S', channel="C3",
                    interface="notebook"):
    time = read_traces_index_trc(tmstps, path, strftime, channel)
    scope_traces = initialize_traces(tmstps, time)
    if scope_traces is None:
        return None
    for tmstp in custom_tqdm(tmstps, interface, 'load scope traces', leave=True):
        try:
            trace = read_scope_trace_trc_values(tmstp, path, strftime, channel)
        except ValueError:
            continue
        scope_traces.loc[{'tmstp': tmstp}] = trace
    return scope_traces


# Read images


def read_image(tmstp: pd.Timestamp, path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S'):
    fits_path = path / 'FITS Files'
    file = fits_path / tmstp.strftime(strftime + '_full.fts')
    if file.is_file():
        with astropy.io.fits.open(file) as fits_file:
            data = fits_file[0].data
            return np.transpose(data, axes=[0, 2, 1])


def read_image_coords(image, sensor_widths: Tuple[int, int] = (1100, 214)):
    n_images, n_pixel_x, n_pixel_y = image.shape
    sensor_width_x, sensor_width_y = sensor_widths
    sensor_width_x *= 1 - 1 / n_pixel_x
    sensor_width_y *= 1 - 1 / n_pixel_y

    x = np.linspace(-sensor_width_x / 2, sensor_width_x / 2, n_pixel_x)
    y = np.linspace(-sensor_width_y / 2, sensor_width_y / 2, n_pixel_y)
    image_names = ['image_' + str(i).zfill(2) for i in range(n_images)]

    return image_names, x, y


def find_first_image(tmstps: List[pd.Timestamp], path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S'):
    for tmstp in tmstps:
        image = read_image(tmstp, path, strftime)
        return image


def initialize_images(tmstps: List[pd.Timestamp], path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S'):
    try:
        image = find_first_image(tmstps, path, strftime)
        image_names, x, y = read_image_coords(image)
    except AttributeError:
        return None
    shape = (len(tmstps), len(x), len(y))
    empty_image = np.full(shape, np.NaN, dtype=np.float32)

    images = xr.Dataset(
        {name: (['tmstp', 'x', 'y'], empty_image.copy()) for name in image_names},
        coords={'tmstp': tmstps, 'x': x, 'y': y}
    )
    return images


def read_images(tmstps: List[pd.Timestamp], path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S',
                interface: str = "tqdm"):
    images = initialize_images(tmstps, path, strftime)
    if not images:
        return None
    for tmstp in custom_tqdm(tmstps, interface, "Read images...", leave=True):
        image_list = read_image(tmstp, path, strftime)
        if image_list is not None:
            for i, image in enumerate(image_list):
                name = 'image_' + str(i).zfill(2)
                images[name].loc[{'tmstp': tmstp}] = image
    return images


# Get raw data


def read_raw_data(tmstps: List[pd.Timestamp], path: Path, strftime: str = '%Y_%m_%d_%H.%M.%S',
                  csv_kwargs=None, fast_csv_kwargs=None, interface: str = "notebook"):
    raw_data = xr.Dataset()
    images = read_images(tmstps, path, strftime, interface)
    if images is not None:
        raw_data = xr.merge([raw_data, images])
    traces = read_traces(tmstps, path, strftime, csv_kwargs, fast_csv_kwargs, interface)
    if traces is not None:
        raw_data['scope_traces'] = traces
    raw_data["parameters"] = read_parameters(tmstps, path, strftime, **csv_kwargs)
    return raw_data


def raw_data_to_multiindex(data: xr.Dataset):
    parameters = data.parameters.to_pandas()
    data = data.drop_vars("parameters")
    data = data.drop_dims("param_dim")
    get_unique_parameters(parameters)
    multi_index = get_shot_multiindex(parameters)
    new_indices = multi_index.to_frame().set_index('tmstp')
    data = data.merge(new_indices)
    return data.set_index(shot=('tmstp', *new_indices.columns))


def get_unique_parameters(parameters):
    _parameters = parameters
    parameters = _parameters.iloc[0].loc[_parameters.nunique() == 1]
    return parameters


def get_shot_multiindex(parameters):
    variables = parameters.loc[:, parameters.nunique() > 1]
    return pd.MultiIndex.from_frame(variables.reset_index())
