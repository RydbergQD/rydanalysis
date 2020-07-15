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
        self.sensor_widths = sensor_widths
        self.handle_key_errors = handle_key_errors

        try:
            self.scope_trace_index = self._get_scope_traces_index()
            self.has_traces = True
        except AttributeError:
            self.has_traces = False

        try:
            self.n_images, self.image_coords_x, self.image_coords_y = self._get_image_coords()
            self.has_images = True
        except AttributeError:
            self.has_images = False

        self.raw_data = self.get_raw_data(self.tmstps)

    @property
    def tmstps(self):
        variables_path = self.path / 'Variables'

        tmstps = []
        for path in variables_path.glob(self.filename_pattern + '.txt'):
            try:
                tmstps.append(pd.to_datetime(path.name, format=self.strftime + '.txt'))
            except:
                print("couldn't read {0}. Skipping this file...".format(path.name))
                pass
        return tmstps

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

    def _get_image_coords(self):
        images_path = self.path / 'FITS Files'
        for file in images_path.glob(self.filename_pattern + '_full.fts'):
            if file.is_file():
                with astropy.io.fits.open(file) as fits_file:
                    image = fits_file[0].data

                n_images, n_pixel_x, n_pixel_y = image.shape

                sensor_width_x, sensor_width_y = self.sensor_widths
                sensor_width_x *= 1 - 1 / n_pixel_x
                sensor_width_y *= 1 - 1 / n_pixel_y
                # pixel_length = sensor_width / n_pixel
                # sensor_width -= pixel_length
                x = np.linspace(-sensor_width_x / 2, sensor_width_x / 2, n_pixel_x)
                y = np.linspace(-sensor_width_y / 2, sensor_width_y / 2, n_pixel_y)

                return n_images, x, y
        raise AttributeError('No images detected')

    def initialize_images(self, tmstps):
        x = self.image_coords_x
        y = self.image_coords_y
        n_images = self.n_images
        shape = (len(tmstps), len(x), len(y))
        empty_image = np.full(shape, np.NaN, dtype=np.float32)

        images = xr.Dataset(
            {
                'image_' + str(i).zfill(2): (['shot', 'x', 'y'], empty_image.copy()) for i in range(n_images)
            },
            coords={
                'shot': self.get_shot_multi_index(tmstps),
                'x': x,
                'y': y}
        )

        return images

    def get_images(self, tmstps):
        fits_path = self.path / 'FITS Files'
        images = self.initialize_images(tmstps)
        for n, tmstp in enumerate(tqdm(tmstps, desc='load images', leave=False)):
            file = fits_path / tmstp.strftime(self.strftime + '_full.fts')
            if file.is_file():
                with astropy.io.fits.open(file) as fits_file:
                    image_list = fits_file[0].data
                    for i, image in enumerate(image_list):
                        images['image_' + str(i).zfill(2)].loc[{'tmstp': tmstp}] = image
        return images

    def _get_scope_traces_index(self):
        scope_traces_path = self.path / 'Scope Traces'
        for path in scope_traces_path.glob(self.filename_pattern + '_C1.csv'):
            trace = pd.read_csv(path, **self.csv_kwargs)
            if trace is not None:
                return trace.index
        raise AttributeError("No scope_traces are found.")

    def initialize_traces(self, tmstps):
        time = self.scope_trace_index
        shape = (len(tmstps), time.size)

        scope_traces = xr.DataArray(
            np.full(shape, np.NaN, dtype=np.float32),
            dims=['shot', 'time'],
            coords={'shot': self.get_shot_multi_index(tmstps), 'time': time.values}
        )
        return scope_traces

    def get_scope_trace(self, tmstp):
        file = self.path / 'Scope Traces' / tmstp.strftime(self.strftime + '_C1.csv')
        if file.is_file():
            scope_trace = pd.read_csv(file, **self.fast_csv_kwargs)
            if scope_trace.shape[0] > 1:
                return scope_trace.values

    def get_scope_traces(self, tmstps):
        scope_traces = self.initialize_traces(tmstps)
        for tmstp in tqdm(tmstps, desc='load scope traces', leave=False):
            scope_traces.loc[{'tmstp': tmstp}] = self.get_scope_trace(tmstp)
        return scope_traces

    def get_raw_data(self, tmstps):
        raw_data = xr.Dataset()
        if self.has_images:
            images = self.get_images(tmstps)
            raw_data = xr.merge([raw_data, images])
        if self.has_traces:
            scope_traces = self.get_scope_traces(tmstps)
            raw_data['scope_traces'] = scope_traces
        return raw_data

    def update_raw_data(self):
        tmstps = set(self.tmstps)
        tmstps.difference_update(self.raw_data.tmstp.values)
        new_raw_data = self.get_raw_data(tmstps)
        self.raw_data = xr.concat([self.raw_data, new_raw_data], dim='shot')

    @property
    def data(self):
        raw_data = self.raw_data
        _parameters = raw_data.shot.reset_index('shot').to_dataframe().drop('shot', axis=1)
        variables = _parameters.loc[:, _parameters.nunique() > 1]
        parameters = _parameters.loc[0, _parameters.nunique() == 1]
        raw_data['shot'] = pd.MultiIndex.from_frame(variables)
        raw_data.attrs.update(parameters)
        return raw_data
