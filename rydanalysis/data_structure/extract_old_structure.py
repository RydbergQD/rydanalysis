from prefect import task, Flow, Parameter, unmapped
import pandas as pd
import astropy.io.fits
import numpy as np
import xarray as xr
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from tqdm.notebook import tqdm
import datetime
from distutils.dir_util import copy_tree


@dataclass
class CSVParams:
    name = "csv params"
    index_col: int = 0
    squeeze: bool = True
    sep: str = '\t'
    decimal: str = ','
    header: any = None


@dataclass
class FastCSVParams:
    name = "csv params"
    usecols: List[int] = field(default_factory=lambda: [1])
    squeeze: bool = True
    sep: str = '\t'
    decimal: str = ','
    header: any = None


@dataclass
class OldStructure:
    path: Path
    strftime: str = '%Y_%m_%d_%H.%M.%S'
    date_strftime: str = '%Y_%m_%d'
    filename_pattern: str = '????_??_??_??.??.??'
    csv_kwargs: dataclass = CSVParams()
    fast_csv_kwargs: dataclass = FastCSVParams()
    handle_key_errors: str = 'ignore'
    sensor_widths: Tuple = (1100, 214)
    batch_size: int = field(default=500)

    def __post_init__(self):
        self.path = Path(self.path)

    def copy_sequences_variables(self, destiny_path):
        origin_path = self.path
        for dir_name in ('Experimental Sequences', 'Variables'):
            (destiny_path / dir_name).mkdir(exist_ok=True)
            copy_tree(
                str(origin_path / dir_name),
                str(destiny_path / dir_name)
            )

    @property
    def base_path(self):
        return self.path.parent.parent

    @property
    def scan_name(self):
        try:
            return self.path.parts[-1]
        except IndexError:
            return None

    @property
    def strf_date(self):
        try:
            return self.path.parts[-2]
        except IndexError:
            return None

    @property
    def date(self):
        if self.strf_date is None:
            return None
        return datetime.datetime.strptime(self.strf_date, self.date_strftime).date()

    def is_dir(self):
        if not self.path.is_dir():
            raise AttributeError("The given path is no directory. Check the path!")
        return True

    def extract_tmstps(self):
        path = self.path / 'Variables'
        tmstps = []
        for sub_path in path.glob(self.filename_pattern + ".txt"):
            try:
                tmstps.append(pd.to_datetime(sub_path.name, format=self.strftime + '.txt'))
            except ValueError:
                print("couldn't read {0}. Skipping this file...".format(sub_path.name))
        return tmstps

    def extract_parameters(self, tmstp):
        parameter_path = self.path / 'Variables' / tmstp.strftime(self.strftime + '.txt')
        parameters = pd.read_csv(parameter_path, **asdict(self.csv_kwargs))
        parameters.name = tmstp
        parameters.index.name = None
        parameters.drop('dummy', inplace=True)
        return parameters

    def reduce_parameters(self, tmstps):
        _parameters = [self.extract_parameters(tmstp) for tmstp in tmstps]

        _parameters = pd.concat(_parameters, axis=1)
        _parameters = _parameters.T
        _parameters.index.name = 'tmstp'
        _parameters.columns.name = "param_dim"
        return _parameters

    def extract_image(self, tmstp):
        fits_path = self.path / 'FITS Files'
        file = fits_path / tmstp.strftime(self.strftime + '_full.fts')
        if file.is_file():
            with astropy.io.fits.open(file) as fits_file:
                data = fits_file[0].data
                return np.transpose(data, axes=[0, 2, 1])

    def _get_image_coords(self, image):
        n_images, n_pixel_x, n_pixel_y = image.shape
        sensor_width_x, sensor_width_y = self.sensor_widths
        sensor_width_x *= 1 - 1 / n_pixel_x
        sensor_width_y *= 1 - 1 / n_pixel_y

        x = np.linspace(-sensor_width_x / 2, sensor_width_x / 2, n_pixel_x)
        y = np.linspace(-sensor_width_y / 2, sensor_width_y / 2, n_pixel_y)
        image_names = ['image_' + str(i).zfill(2) for i in range(n_images)]

        return image_names, x, y

    def reduce_images(self, tmstps):
        images = [self.extract_image(t) for t in tqdm(tmstps, desc='load images', leave=False)]
        images = list(filter(None.__ne__, images))
        if not images:
            return None
        names, x, y = self._get_image_coords(images[0])
        return xr.DataArray(images, coords=dict(tmstp=tmstps, names=names, x=x, y=y),
                            dims=("tmstp", "names", "x", "y")).to_dataset("names")

    def extract_scope_trace(self, tmstp, method):
        traces_path = self.path / 'Scope Traces'
        file = traces_path / tmstp.strftime(self.strftime + '_C1.csv')
        if not file.is_file():
            pass
        try:
            if method == 'fast':
                scope_trace = pd.read_csv(file, **asdict(self.fast_csv_kwargs))
                if scope_trace.shape[0] > 1:
                    scope_trace.name = tmstp
                    return scope_trace
            else:
                return pd.read_csv(file, **asdict(self.csv_kwargs))
        except:
            pass

    def _get_scope_traces_index(self, tmstps):
        for tmstp in tmstps:
            trace = self.extract_scope_trace(tmstp, method='slow')
            if trace is not None:
                return trace.index

    def initialize_traces(self, tmstps):
        time = self._get_scope_traces_index(tmstps)
        if time is None:
            return None
        shape = (len(tmstps), time.size)

        scope_traces = xr.DataArray(
            np.full(shape, np.NaN, dtype=np.float32),
            dims=['tmstp', 'time'],
            coords={'tmstp': tmstps, 'time': time.values}
        )
        return scope_traces

    def reduce_traces(self, tmstps):
        scope_traces = self.initialize_traces(tmstps)
        if scope_traces is None:
            return None
        for tmstp in tqdm(tmstps, desc='load scope traces', leave=False):
            trace = self.extract_scope_trace(tmstp, method='fast')
            scope_traces.loc[{'tmstp': tmstp}] = trace
        return scope_traces

    def get_raw_data(self, tmstps):
        parameters = self.reduce_parameters(tmstps)
        images = self.reduce_images(tmstps)
        traces = self.reduce_traces(tmstps)

        raw_data = xr.Dataset()
        raw_data["parameters"] = parameters
        if traces is not None:
            raw_data['scope_traces'] = traces
        if images is not None:
            raw_data = xr.merge([raw_data, images])
        return raw_data

    def save_data(self, path):
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True)

        if self.batch_size:
            if self.batch_size <= 0:
                raise ValueError("batch size should be positive or None. ")

        tmstps = self.extract_tmstps()
        bar = tqdm(total=len(tmstps))
        i = 0
        while True:
            tmstp_batch = tmstps[:self.batch_size]

            raw_data = self.get_raw_data(tmstp_batch)
            raw_data.to_netcdf(path / ("raw_data_batch" + str(i).zfill(3) + ".h5"))

            bar.update(len(tmstp_batch))
            tmstps = [t for t in tmstps if t not in tmstp_batch]
            i += 1
            if not tmstps:
                break

        bar.close()

    @property
    def data(self):
        tmstps = self.extract_tmstps()
        if len(tmstps) > self.batch_size:
            raise ResourceWarning("Your dataset is larger than the batch_size. Consider saving"
                                  " the data using 'save data' or increase the batch_size.")
        raw_data = self.get_raw_data(tmstps)
        return raw_data_to_multiindex(raw_data)


def raw_data_to_multiindex(data):
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


def load_data(path, lazy=False):
    path = Path(path)
    with xr.open_mfdataset(path.glob("*.h5"), concat_dim="tmstp") as data:
        data = raw_data_to_multiindex(data)
        if not lazy:
            return data.load()
        else:
            return data
