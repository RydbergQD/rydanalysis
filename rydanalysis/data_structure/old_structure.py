import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr
import numpy as np
from pathlib import Path
import astropy.io.fits


class OldStructure:
    raw_data: xr.Dataset
    strftime = '%Y_%m_%d_%H.%M.%S'
    filename_pattern = '????_??_??_??.??.??'
    csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)

    def __init__(self, path, handle_key_errors='ignore', sensor_widths=(214, 1100)):
        self.path = Path(path)
        self.is_dir()
        self.sensor_widths = sensor_widths
        self.handle_key_errors = handle_key_errors

        self.tmstps = []
        self.parameters = pd.DataFrame()
        self.images = None
        self.traces = None

        self.update()

    def is_dir(self):
        if not self.path.is_dir():
            raise AttributeError("The given path is no directory. Check the path!")
        return True

    def update_tmstps(self):
        variables_path = self.path / 'Variables'
        new_tmstps = [tmstp for tmstp in self.iter_tmstps(variables_path, self.filename_pattern + '.txt')
                      if tmstp not in self.tmstps]
        self.tmstps.extend(new_tmstps)
        return new_tmstps

    def iter_tmstps(self, path: Path, pattern: str):
        for sub_path in path.glob(pattern):
            try:
                tmstp = pd.to_datetime(sub_path.name, format=self.strftime + '.txt')
                yield tmstp
            except ValueError:
                print("couldn't read {0}. Skipping this file...".format(sub_path.name))

    def get_single_parameters(self, tmstp):
        parameter_path = self.path / 'Variables' / tmstp.strftime(self.strftime + '.txt')
        parameters = pd.read_csv(parameter_path, **self.csv_kwargs)
        parameters.name = tmstp
        parameters.index.name = None
        parameters.drop('dummy', inplace=True)
        return parameters

    def get_parameters(self, tmstps):
        _parameters = pd.concat(
            [self.get_single_parameters(tmstp)
             for tmstp in tqdm(tmstps, desc='find parameters and variables', leave=False)],
            axis=1
        )
        _parameters = _parameters.T
        _parameters.index.name = 'tmstp'
        return _parameters

    def get_shot_multi_index(self, tmstps):
        shot_multi_index = self.get_parameters(tmstps).reset_index()
        return pd.MultiIndex.from_frame(shot_multi_index)

    def save_raw_data(self, path=None):
        if path is None:
            path = self.path
        data = self.raw_data
        data = data.reset_index('shot')
        data.to_netcdf(path / 'raw_data.h5')

    def get_old_la(self):
        old_la_path = self.path / 'FITS Files' / '00_all_fit_results.csv'
        old_la = pd.read_csv(old_la_path, **self.csv_kwargs)
        old_la.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        old_la.time = old_la.time.apply(lambda t: pd.to_datetime(t, unit='s', origin=self.raw_data.tmstps[0].date()))
        old_la.set_index('time', inplace=True)
        del old_la['Index']
        return old_la

    def iter_fits_files(self, tmstps):
        fits_path = self.path / 'FITS Files'
        for n, tmstp in enumerate(tmstps):
            file = fits_path / tmstp.strftime(self.strftime + '_full.fts')
            image = self._fits_to_image(file)
            yield tmstp, image
        # for file in images_path.glob(self.filename_pattern + '_full.fts'):

    @staticmethod
    def _fits_to_image(file):
        if file.is_file():
            with astropy.io.fits.open(file) as fits_file:
                return fits_file[0].data

    def _get_image_coords(self):
        if self.images:
            return self.images.data_vars.keys(), self.images.x, self.images.y,

        for tmstp, image in self.iter_fits_files(self.tmstps):
            n_images, n_pixel_x, n_pixel_y = image.shape

            sensor_width_x, sensor_width_y = self.sensor_widths
            sensor_width_x *= 1 - 1 / n_pixel_x
            sensor_width_y *= 1 - 1 / n_pixel_y

            x = np.linspace(-sensor_width_x / 2, sensor_width_x / 2, n_pixel_x)
            y = np.linspace(-sensor_width_y / 2, sensor_width_y / 2, n_pixel_y)
            image_names = ['image_' + str(i).zfill(2) for i in range(n_images)]

            return image_names, x, y
        raise AttributeError('No images found')

    def initialize_images(self, tmstps):
        try:
            image_names, x, y = self._get_image_coords()
        except AttributeError:
            return None
        shape = (len(tmstps), len(x), len(y))
        empty_image = np.full(shape, np.NaN, dtype=np.float32)

        images = xr.Dataset(
            {name: (['tmstp', 'x', 'y'], empty_image.copy()) for name in image_names},
            coords={'tmstp': tmstps, 'x': x, 'y': y}
        )
        return images

    def get_images(self, tmstps):
        images = self.initialize_images(tmstps)
        if not images:
            return None
        for tmstp, image_list in self.iter_fits_files(tmstps):
            for i, image in enumerate(image_list):
                name = 'image_' + str(i).zfill(2)
                images[name].loc[{'tmstp': tmstp}] = image
        return images

    def _iter_traces(self, tmstps, method='fast'):
        traces_path = self.path / 'Scope Traces'
        for tmstp in tmstps:
            file = traces_path / tmstp.strftime(self.strftime + '_C1.csv')
            if file.is_file():
                yield tmstp, self.read_scope_trace(file, method)

    def read_scope_trace(self, file, method):
        if method == 'fast':
            scope_trace = pd.read_csv(file, **self.fast_csv_kwargs)
            if scope_trace.shape[0] > 1:
                return scope_trace.values
        else:
            return pd.read_csv(file, **self.csv_kwargs)

    def _get_scope_traces_index(self):
        if self.traces:
            return self.traces['time']

        for tmstp, trace in self._iter_traces(self.tmstps, method='slow'):
            if trace is not None:
                return trace.index

    def initialize_traces(self, tmstps):
        time = self._get_scope_traces_index()
        if not time:
            return None
        shape = (len(tmstps), time.size)

        scope_traces = xr.DataArray(
            np.full(shape, np.NaN, dtype=np.float32),
            dims=['tmstp', 'time'],
            coords={'tmstp': tmstps, 'time': time.values}
        )
        return scope_traces

    def get_scope_traces(self, tmstps):
        scope_traces = self.initialize_traces(tmstps)
        if not scope_traces:
            return None
        for tmstp, trace in tqdm(self._iter_traces(tmstps, 'fast'), desc='load scope traces', leave=False):
            scope_traces.loc[{'tmstp': tmstp}] = trace
        return scope_traces

    def update_images(self, images):
        if self.images:
            self.images = xr.concat([self.images, images], dim='tmstp')
        else:
            self.images = images

    def update_traces(self, traces):
        if self.traces:
            self.traces = xr.concat([self.traces, traces], dim='tmstp')
        else:
            self.traces = traces

    def update(self):
        tmstps = self.update_tmstps()
        if not tmstps:
            return

        images = self.get_images(tmstps)
        self.update_images(images)

        traces = self.get_scope_traces(tmstps)
        self.update_traces(traces)

        parameters = self.get_parameters(tmstps)
        self.parameters = pd.concat([self.parameters, parameters], axis=0)

    def get_raw_data(self):
        raw_data = xr.Dataset()
        if self.images:
            raw_data = xr.merge([raw_data, self.images])
        if self.traces:
            raw_data['scope_traces'] = self.traces
        return raw_data

    @property
    def shot_multiindex(self):
        _parameters = self.parameters
        variables = _parameters.loc[:, _parameters.nunique() > 1]
        return pd.MultiIndex.from_frame(variables.reset_index())

    @property
    def unique_parameters(self):
        _parameters = self.parameters
        parameters = _parameters.iloc[0].loc[_parameters.nunique() == 1]
        return parameters

    @property
    def data(self):
        raw_data = self.get_raw_data()
        raw_data = raw_data.reindex(tmstp=self.shot_multiindex)
        raw_data = raw_data.rename(tmstp='shot')
        raw_data.attrs.update(self.unique_parameters)
        return raw_data
