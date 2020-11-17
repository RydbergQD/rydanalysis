import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
import lecroyparser
from rydanalysis.auxiliary.user_input import custom_tqdm
from abc import ABC, abstractmethod


# Read scope traces


def read_trace_csv(
    tmstp: pd.Timestamp, path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S", **csv_kwargs
):
    traces_path = path / "Scope Traces"
    file = traces_path / tmstp.strftime(strftime + "_C1.csv")
    if not file.is_file():
        pass
    try:
        return pd.read_csv(file, **csv_kwargs)
    except OSError:
        pass


def read_trace_csv_values(
    tmstp: pd.Timestamp, path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S", **csv_kwargs
):
    scope_trace = read_trace_csv(tmstp, path, strftime, **csv_kwargs)
    if scope_trace is None:
        return None
    if scope_trace.shape[0] > 1:
        scope_trace.name = tmstp
        return scope_trace


def read_trace_trc(tmstp, path, strftime: str = "%Y_%m_%d_%H.%M.%S", channel="C3"):
    traces_path = path / "Scope Traces"
    file = traces_path / tmstp.strftime(channel + strftime + "00000.trc")
    try:
        return lecroyparser.ScopeData(str(file))
    except FileNotFoundError:
        return None


def read_trace_trc_values(
    tmstp, path, strftime: str = "%Y_%m_%d_%H.%M.%S", channel="C3"
):
    data = read_trace_trc(tmstp, path, strftime, channel)
    if data is not None:
        return data.y.astype(np.float16)


def find_valid_trace(path):
    traces_path = path / "Scope Traces"
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
        dims=["tmstp", "time"],
        coords={"tmstp": tmstps, "time": time},
    )
    return scope_traces


class TraceReader(ABC):
    def __init__(
        self,
        tmstps,
        path: Path,
        times=None,
        strftime: str = "%Y_%m_%d_%H.%M.%S",
        interface="notebook",
    ):
        self.path = path
        self.tmstps = tmstps
        self.strftime = strftime
        if times is None:
            times = self.read_traces_index(tmstps)
        self.times = times
        self.interface = interface

    def read_traces(self):
        scope_traces = initialize_traces(self.tmstps, self.times)
        if scope_traces is None:
            return None
        for tmstp in custom_tqdm(
            self.tmstps, self.interface, "load scope traces", leave=True
        ):
            try:
                trace = self.read_trace_values(tmstp)
            except ValueError:
                continue
            scope_traces.loc[{"tmstp": tmstp}] = trace
        return scope_traces

    def read_traces_index(self, tmstps):
        for tmstp in tmstps:
            trace = self.read_trace(tmstp)
            if trace is not None:
                return self.get_index(trace)

    @abstractmethod
    def read_trace(self, tmstp):
        pass

    @staticmethod
    @abstractmethod
    def get_index(trace):
        pass

    @abstractmethod
    def read_trace_values(self, tmstp):
        pass


class TraceReaderTRC(TraceReader):
    def __init__(
        self,
        tmstps,
        path: Path,
        times=None,
        channel=None,
        strftime: str = "%Y_%m_%d_%H.%M.%S",
        interface="notebook",
    ):
        if channel is None:
            _, channel = find_valid_trace(path)
        self.channel = channel

        super(TraceReaderTRC, self).__init__(tmstps, path, times, strftime, interface)

    def read_trace(self, tmstp):
        return read_trace_trc(tmstp, self.path, self.strftime, self.channel)

    def read_trace_values(self, tmstp):
        return read_trace_trc_values(
            tmstp, self.path, self.strftime, self.channel
        )

    @staticmethod
    def get_index(trace):
        return trace.index.values


class TraceReaderCSV(TraceReader):
    def __init__(
        self,
        tmstps,
        path: Path,
        times=None,
        csv_kwargs=None,
        fast_csv_kwargs=None,
        strftime: str = "%Y_%m_%d_%H.%M.%S",
        interface="notebook",
    ):
        if csv_kwargs is None:
            csv_kwargs = dict(
                index_col=0, squeeze=True, sep="\t", decimal=",", header=None
            )
        self.csv_kwargs = csv_kwargs
        if fast_csv_kwargs is None:
            fast_csv_kwargs = dict(
                usecols=[1], squeeze=True, sep="\t", decimal=",", header=None
            )
        self.fast_csv_kwargs = fast_csv_kwargs

        super(TraceReaderCSV, self).__init__(tmstps, path, times, strftime, interface)

    def read_trace(self, tmstp):
        return read_trace_csv(tmstp, self.path, self.strftime, **self.csv_kwargs)

    def read_trace_values(self, tmstp):
        return read_trace_csv_values(
            tmstp, self.path, self.strftime, **self.fast_csv_kwargs
        )

    @staticmethod
    def get_index(trace):
        return trace.x


def read_traces(
    tmstps,
    path: Path,
    times=None,
    strftime: str = "%Y_%m_%d_%H.%M.%S",
    csv_kwargs=None,
    fast_csv_kwargs=None,
    interface="notebook",
):
    method, channel = find_valid_trace(Path(path))
    if method == "csv":
        trace_reader = TraceReaderCSV(
            tmstps, path, times, csv_kwargs, fast_csv_kwargs, strftime, interface
        )
    else:
        trace_reader = TraceReaderTRC(tmstps, path, times, channel, strftime, interface)
    return trace_reader.read_traces()
