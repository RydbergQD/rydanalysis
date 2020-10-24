import datetime
import os
from dataclasses import dataclass, field
from distutils.dir_util import copy_tree
from shutil import copy
from typing import Dict, Optional, Iterable

from ..auxiliary.user_input import custom_tqdm, user_input, choose_from_options
from .old_structure_methods import *
import streamlit as st


@dataclass
class OldStructure:
    path: str = "."
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

    @property
    def _path(self):
        return Path(self.path)

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

    @property
    def base_path(self):
        if self.strf_date:
            return self._path.parent.parent
        else:
            return self._path.parent

    @property
    def scan_name(self):
        try:
            return self._path.parts[-1]
        except IndexError:
            return ""

    @property
    def strf_date(self):
        try:
            return self._path.parts[-2]
        except IndexError:
            return None

    @property
    def date(self):
        if self.strf_date is None:
            return None
        return datetime.datetime.strptime(self.strf_date, self.date_strftime).date()

    def is_dir(self):
        if not self._path.is_dir():
            raise AttributeError("The given _path is no directory. Check the _path!")
        return True

    def extract_tmstps(self):
        path = self._path / 'Variables'
        tmstps = []
        for sub_path in path.glob(self.filename_pattern + ".txt"):
            try:
                tmstps.append(pd.to_datetime(sub_path.name, format=self.strftime + '.txt'))
            except ValueError:
                print("couldn't read {0}. Skipping this file...".format(sub_path.name))
        return tmstps

    def read_parameters(self, tmstp):
        return read_parameters(tmstp, self._path, self.strftime, **self.csv_kwargs)

    def get_parameters(self, tmstps):
        return read_parameters(tmstps, self._path, self.strftime, **self.csv_kwargs)

    def read_image(self, tmstp):
        return read_image(tmstp, self._path, self.strftime)

    def get_images(self, tmstps):
        return get_images(tmstps, self._path, self.strftime, self.interface)

    def read_scope_trace(self, tmstp):
        return read_scope_trace(tmstp, self._path, self.strftime, **self.csv_kwargs)

    def get_traces(self, tmstps):
        return get_traces(tmstps, self._path, self.strftime, self.csv_kwargs, self.fast_csv_kwargs,
                          self.interface)

    def get_raw_data(self, tmstps):
        return get_raw_data(tmstps, self._path, self.strftime, self.csv_kwargs, self.fast_csv_kwargs,
                            self.interface)

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
            old_tmstps = get_existing_tmstps(destiny_path)
            tmstps = compare_tmstps(tmstps, old_tmstps)
        else:
            for f in destiny_path.glob('*.h5'):
                os.remove(f)

        if not destiny_path.is_dir():
            destiny_path.mkdir(parents=True)

        chunks = self.get_chunks(tmstps)
        for chunk in custom_tqdm(chunks, self.interface, "Iterate chunks", leave=True):
            raw_data = self.get_raw_data(chunk)
            name = pd.to_datetime(str(chunk[0]))
            name = name.strftime(self.strftime)
            raw_data.to_netcdf(destiny_path / ("raw_data_batch" + name + ".h5"))

    @property
    def data(self):
        tmstps = self.extract_tmstps()
        if len(tmstps) > self.chunk_size:
            raise ResourceWarning("Your dataset is larger than the batch_size. Consider saving"
                                  " the data using 'save data' or increase the batch_size.")
        raw_data = self.get_raw_data(tmstps)
        return raw_data_to_multiindex(raw_data)


def get_existing_tmstps(destiny_path: Path) -> Iterable[pd.DatetimeIndex]:
    try:
        with xr.open_mfdataset(destiny_path.glob("*.h5"), concat_dim="tmstp") as data:
            tmstps = map(pd.to_datetime, data.tmstp.values)
            return tmstps
    except OSError:
        return []


def compare_tmstps(new_tmstps, old_tmstps):
    new_tmstps = set(new_tmstps)
    return list(new_tmstps.difference(old_tmstps))


def load_data(path, lazy=False, to_multiindex=True):
    path = Path(path)
    with xr.open_mfdataset(path.glob("*.h5"), concat_dim="tmstp") as data:
        if not lazy:
            data = data.load()
    if to_multiindex:
        data = raw_data_to_multiindex(data)
    return data
