import datetime
import os
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from shutil import copy
from typing import Dict, Optional, Union, Tuple

import pandas as pd

from rydanalysis.auxiliary.user_input import custom_output, custom_tqdm
from .extract_data import read_tmstps_txt, read_parameters, read_parameters_single, read_raw_data, \
    read_tmstps_h5, compare_tmstps, raw_data_to_multiindex
from .read_images import read_images, read_image
from .read_traces import read_traces, read_trace_csv


@dataclass
class OldStructure:
    path: InitVar[Union[str, Path]] = "."
    base_path: Optional[Union[str, Path]] = None
    date: Optional[datetime.date] = None
    scan_name: Optional[str] = None
    strftime: str = '%Y_%m_%d_%H.%M.%S'
    date_strftime: str = '%Y_%m_%d'
    filename_pattern: str = '????_??_??_??.??.??'
    csv_kwargs: Dict = field(
        default_factory=lambda: dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    )
    fast_csv_kwargs: Dict = field(
        default_factory=lambda: dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)
    )
    handle_key_errors: str = 'ignore'
    sensor_widths: Tuple = (1100, 214)
    chunk_size: Optional[int] = field(default=500)
    interface: str = "notebook"
    export_path: str = r"\\147.142.18.81\qd-local\qd\rydberg\Projekte - Projects\2020_Aging"
    date_to_destiny: bool = True
    append_to_old_data = True

    def __post_init__(self, path: Union[str, Path]):
        if self.base_path is not None:
            pass
        path = Path(path).absolute()
        self.scan_name = path.parts[-1]
        self.date = get_date_from_path(path, self.date_strftime)
        if self.date is not None:
            self.base_path = path.parent.parent
        else:
            self.base_path = path.parent

    @property
    def _path(self):
        if self.date is not None:
            return self.base_path / self.strf_date / self.scan_name
        else:
            return self.base_path / self.scan_name

    @_path.setter
    def _path(self, path):
        self.__post_init__(path)

    def set_base_path(self, path: Union[str, Path]):
        path = Path(path)
        if path == self.base_path:
            pass
        self.base_path = path
        self.check_scan_name()

    def set_date(self, date):
        if date == self.date:
            pass
        self.date = date
        self.check_scan_name()

    def check_scan_name(self):
        if self.scan_name not in self.scan_names:
            self.scan_name = None

    @property
    def strf_date(self):
        if self.date is not None:
            return self.date.strftime(self.date_strftime)

    @strf_date.setter
    def strf_date(self, value: Optional[str]):
        if value == self.strf_date:
            pass

        if value is None:
            self.date = datetime.datetime.strptime(self.strf_date, self.date_strftime).date()

    @property
    def scan_names(self):
        if self.date is not None:
            path = self.base_path / self.strf_date
        else:
            path = self.base_path
        if path.is_dir():
            return [x.name for x in path.iterdir() if x.is_dir()]
        else:
            custom_output(
                'No experimental run was found on that date. Consider Changing the base path '
                'or verify the date.', interface=self.interface)
            return []

    @property
    def scan_name_index(self):
        if self.scan_name in self.scan_names:
            return self.scan_names.index(self.scan_name)
        else:
            return 0

    def is_dir(self):
        if not self._path.is_dir():
            raise AttributeError("The given _path is no directory. Check the _path!")
        return True

    def get_old_la(self):
        tmstps = self.extract_tmstps()
        initial_time = tmstps[0].date()
        old_la_path = self._path / 'FITS Files' / '00_all_fit_results.csv'
        old_la = pd.read_csv(old_la_path, **self.csv_kwargs)
        old_la.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        old_la.time = old_la.time.apply(
            lambda t: pd.to_datetime(t, unit='s', origin=initial_time))
        old_la.set_index('time', inplace=True)
        del old_la['Index']
        return old_la

    def copy_sequences_variables(self, destiny_path):
        origin_path = self._path
        for dir_name in ('Experimental Sequences', 'Voltages'):
            origin_folder = origin_path / dir_name
            destiny_folder = destiny_path / dir_name
            if not destiny_folder.is_dir():
                destiny_folder.mkdir(exist_ok=True)
            origin_files = set(path.name for path in origin_folder.glob("*.xml"))
            existing_files = set(path.name for path in destiny_folder.glob("*.xml"))
            new_files = list(origin_files - existing_files)
            for file in custom_tqdm(new_files, interface=self.interface, desc="Copy " + dir_name):
                copy(origin_folder / file, destiny_folder)

    def copy_voltages(self, destiny_path):
        path = self._path / "Voltages"
        for file in path.glob(".xml"):
            copy(file, destiny_path / "Voltages")

    def extract_tmstps(self):
        return read_tmstps_txt(self._path, self.filename_pattern, self.strftime)

    def read_parameters_single(self, tmstp):
        return read_parameters_single(tmstp, self._path, self.strftime, **self.csv_kwargs)

    def read_parameters(self, tmstps):
        return read_parameters(tmstps, self._path, self.strftime, **self.csv_kwargs)

    def read_image(self, tmstp):
        return read_image(tmstp, self._path, self.strftime)

    def read_images(self, tmstps):
        return read_images(tmstps, self._path, self.strftime, self.interface)

    def read_trace_csv(self, tmstp):
        return read_trace_csv(tmstp, self._path, self.strftime, **self.csv_kwargs)

    def read_traces(self, tmstps):
        return read_traces(tmstps, self._path, self.strftime, self.csv_kwargs, self.fast_csv_kwargs,
                           self.interface)

    def read_raw_data(self, tmstps):
        return read_raw_data(tmstps, self._path, self.strftime, self.csv_kwargs,
                             self.fast_csv_kwargs, self.interface)

    def get_chunks(self, tmstps):
        if not self.chunk_size:
            return [tmstps]
        if self.chunk_size > 0:
            n = abs(self.chunk_size)
            return [tmstps[i:i + n] for i in range(0, len(tmstps), n)]

    def save_data(self, destiny_path, append=True):
        destiny_path = Path(destiny_path)
        tmstps = self.extract_tmstps()

        if append:
            old_tmstps = read_tmstps_h5(destiny_path)
            tmstps = compare_tmstps(tmstps, old_tmstps)
        else:
            for f in destiny_path.glob('*.h5'):
                os.remove(f)

        if not destiny_path.is_dir():
            destiny_path.mkdir(parents=True)

        chunks = self.get_chunks(tmstps)
        for chunk in custom_tqdm(chunks, self.interface, "Iterate chunks", leave=True):
            raw_data = self.read_raw_data(chunk)
            tmstp = pd.Timestamp(str(chunk[0]))
            name = tmstp.strftime(self.strftime)
            raw_data.to_netcdf(destiny_path / ("raw_data_batch" + name + ".h5"))

    @property
    def data(self):
        tmstps = self.extract_tmstps()
        if len(tmstps) > self.chunk_size:
            raise ResourceWarning("Your dataset is larger than the batch_size. Consider saving"
                                  " the data using 'save data' or increase the batch_size.")
        raw_data = self.read_raw_data(tmstps)
        return raw_data_to_multiindex(raw_data)


def get_date_from_path(path, date_strftime='%Y_%m_%d'):
    path = Path(path)
    if len(path.parts) < 2:
        return None

    strf_date = path.parts[-2]
    try:
        return datetime.datetime.strptime(strf_date, date_strftime).date()
    except ValueError:
        return None
