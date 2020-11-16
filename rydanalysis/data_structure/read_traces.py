import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
import lecroyparser
from rydanalysis.auxiliary.user_input import custom_tqdm


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
