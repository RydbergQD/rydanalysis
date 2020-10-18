import datetime
import os
from dataclasses import dataclass, field
from distutils.dir_util import copy_tree
from typing import Dict, Optional

from ..auxiliary.user_input import custom_tqdm, user_input
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
        for dir_name in ('Experimental Sequences', 'Variables'):
            (destiny_path / dir_name).mkdir(exist_ok=True)
            copy_tree(
                str(origin_path / dir_name),
                str(destiny_path / dir_name)
            )

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

    def save_data(self, destiny_path):
        destiny_path = Path(destiny_path)

        old_tmstps = get_existing_tmstps(destiny_path)
        tmstps = self.extract_tmstps()
        if old_tmstps:
            append_remove = user_input(
                "Found already existing h5 files. Append (a) or overwrite (o)")
            if append_remove == "o":
                for f in destiny_path.glob('*.h5'):
                    os.remove(f)
            else:
                tmstps = compare_tmstps(old_tmstps, tmstps)

        if not destiny_path.is_dir():
            destiny_path.mkdir(parents=True)

        chunks = self.get_chunks(tmstps)
        for chunk in custom_tqdm(chunks, self.interface, "Iterate chunks", leave=True):
            raw_data = self.get_raw_data(chunk)
            raw_data.to_netcdf(destiny_path / ("raw_data_batch" + str(chunk[0]) + ".h5"))

    @property
    def data(self):
        tmstps = self.extract_tmstps()
        if len(tmstps) > self.chunk_size:
            raise ResourceWarning("Your dataset is larger than the batch_size. Consider saving"
                                  " the data using 'save data' or increase the batch_size.")
        raw_data = self.get_raw_data(tmstps)
        return raw_data_to_multiindex(raw_data)


def get_existing_tmstps(destiny_path):
    try:
        data = xr.open_mfdataset(destiny_path / "raw_data*.h5")
        return data.tmstp.values
    except OSError:
        return None


def compare_tmstps(old_tmstps, new_tmstps):
    old_tmstps = set(old_tmstps)
    return list(old_tmstps.difference(new_tmstps))


def load_data(path, lazy=False):
    path = Path(path)
    with xr.open_mfdataset(path.glob("*.h5"), concat_dim="tmstp") as data:
        data = raw_data_to_multiindex(data)
        if not lazy:
            return data.load()
        else:
            return data
