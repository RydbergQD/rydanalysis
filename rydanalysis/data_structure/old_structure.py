from distutils.dir_util import copy_tree

import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr
import numpy as np
from pathlib import Path
import astropy.io.fits
import streamlit as st
import datetime


class OldStructure:
    raw_data: xr.Dataset
    strftime = '%Y_%m_%d_%H.%M.%S'
    date_strftime = '%Y_%m_%d'
    filename_pattern = '????_??_??_??.??.??'
    csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)

    def __init__(self, path, handle_key_errors='ignore', sensor_widths=(1100, 214),
                 initial_update=True):
        self.path = Path(path)
        self.sensor_widths = sensor_widths
        self.handle_key_errors = handle_key_errors

        self.tmstps = []
        self.parameters = pd.DataFrame()
        self.images = None
        self.traces = None

        if initial_update:
            self.is_dir()
            self.update()

    def set_init_kwargs(self):
        handle_key_errors = st.sidebar.text_input(
            "How to handle key errors?", value=self.handle_key_errors)
        sensor_width_x = st.sidebar.number_input("Width of the image in um: ",
                                                 value=self.sensor_widths[0])
        sensor_width_y = st.sidebar.number_input("Height of the image in um: ",
                                                 value=self.sensor_widths[1])
        return handle_key_errors, (sensor_width_x, sensor_width_y)

    def streamlit_from_date(self):
        handle_key_errors, sensor_widths = self.set_init_kwargs()
        base_path = Path(st.text_input('Enter base path', value=str(self.base_path)))
        if not base_path.is_dir():
            st.text("'Base path is not a valid directory. '")
            st.stop()

        # Choose date
        default_date = self.date if self.date else datetime.date.today()
        date: datetime.date = st.date_input('Choose data', default_date)
        strf_date = date.strftime(OldStructure.date_strftime)
        date_path = base_path / strf_date
        if not date_path.is_dir():
            st.text(
                'No experimental run was found on that date. Consider Changing the base path or '
                'verify the date.')
            st.stop()
        scan_names = pd.Series([x.name for x in date_path.iterdir() if x.is_dir()])
        scan_names.insert(0, 'None')
        default_name = self.scan_name if scan_names[self.scan_name].index else 0
        scan_name = st.selectbox('Choose run', scan_names, index=default_name)
        if scan_name == 'None' or scan_name == self.scan_name:
            st.stop()
        path = date_path / scan_name
        self.__init__(path, handle_key_errors=handle_key_errors,
                      sensor_widths=sensor_widths, initial_update=False)

    def streamlit_from_path(self):
        path = st.text_input("Enter path: ", value=str(self.path))
        path = Path(path).absolute()
        if not path.is_dir():
            st.text("'Path is not a valid directory. '")
            st.stop()
        return self.__init__(path)

    def streamlit_update(self):
        if st.button("Load data"):
            self.is_dir()
            self.update()
            return self.data

    def export_data(self, path):
        path = Path(path)
        (path / 'Analysis').mkdir(parents=True)

        self.copy_sequences_variables(path)
        self.data.ryd_data.to_netcdf(path / 'raw_data.h5')

    def streamlit_export(self):
        st.write("""## Export data""")
        if self.date is None:
            export_option = "by path"
        else:
            export_option = st.radio(
                "How to define the destiny folder: ", options=["by date", "by path"]
            )

        if export_option == "by date":
            path = Path(st.text_input("Save as netcdf here"))
            destiny_path = path / self.strf_date / self.scan_name
        else:
            destiny_path = Path(st.text_input("Save as netcdf here"))

        st.text("""Data will be saved here: {}""".format(str(destiny_path)))

        if st.button('to_netcdf'):
            self.export_data(destiny_path)

    def copy_sequences_variables(self, destiny_path):
        origin_path = self.path
        for dir_name in ('Experimental Sequences', 'Variables'):
            (destiny_path / dir_name).mkdir()
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

    def update_tmstps(self):
        variables_path = self.path / 'Variables'
        new_tmstps = [tmstp for tmstp in
                      self.iter_tmstps(variables_path, self.filename_pattern + '.txt')
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
        old_la.time = old_la.time.apply(
            lambda t: pd.to_datetime(t, unit='s', origin=self.raw_data.tmstps[0].date()))
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
                data = fits_file[0].data
                return np.transpose(data, axes=[0, 2, 1])

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
        if time is None:
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
        if scope_traces is None:
            return None
        for tmstp, trace in tqdm(self._iter_traces(tmstps, 'fast'), desc='load scope traces',
                                 leave=False):
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
        if self.images is not None:
            raw_data = xr.merge([raw_data, self.images])
        if self.traces is not None:
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
        multi_index = self.shot_multiindex
        new_indices = multi_index.to_frame().set_index('tmstp')

        raw_data = self.get_raw_data()
        raw_data = raw_data.merge(new_indices)
        raw_data = raw_data.set_index(shot=('tmstp', *new_indices.columns))
        raw_data.attrs.update(self.unique_parameters)
        return raw_data
