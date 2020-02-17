from rydanalysis.IO.os import Directory
from rydanalysis.IO.io import GetterWithTimestamp, _load_path
from rydanalysis.IO.exp_sequence import ExpSequence
from rydanalysis.IO.fits import FitsFile
from rydanalysis.auxiliary.decorators import cached_property

import pandas as pd
from os.path import basename, join, isdir
from tqdm.notebook import tqdm
import warnings
import xarray as xr
import numpy as np
import click
import shutil


class OldStructure(Directory):
    strftime = '%Y_%m_%d_%H.%M.%S'
    csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)

    def __init__(self, path, handle_key_errors='ignore', sensor_widths=(428, 2191.36)):
        super(OldStructure, self).__init__(path)
        self.handle_key_errors = handle_key_errors
        self.tmstps = self.get_tmstps()
        self.sensor_widths = sensor_widths
        self.variables, self.parameters = self._find_variables()

    def _find_variables(self):
        _parameters = pd.concat(
            [self.single_parameters[tmstp]
             for tmstp in tqdm(self.tmstps, desc='find parameters and variables', leave=False)],
            axis=1
        )
        variables = _parameters[_parameters.T.nunique() > 1].T
        variables.index.name = 'tmstp'
        parameters = _parameters[_parameters.T.nunique() == 1].mean(axis=1)
        return variables, parameters

    def __getitem__(self, key):
        path = join(self.path, key)
        try:
            return _load_path(path)
        except KeyError as err:
            if self.handle_key_errors is 'ignore':
                pass
            elif self.handle_key_errors is 'warning':
                warnings.warn(str(err) + ' not found')
            else:
                raise err

    def __repr__(self):
        return "Old structure: " + self.path

    def __str__(self):
        return "Old structure: " + self.__name__

    def get_tmstps(self):
        tmstps = []
        for path in self['Variables'].iter_files():
            try:
                tmstps.append(pd.to_datetime(basename(path), format=self.strftime + '.txt'))
            except:
                print("couldn't read {0}. Skipping this file...".format(basename(path)))
                pass
        return tmstps

    def get_image_coords(self, image, index):
        n_pixel = image.shape[index]
        sensor_width = self.sensor_widths[index]
        pixel_length = sensor_width / n_pixel
        sensor_width -= pixel_length

        return np.linspace(-sensor_width / 2, sensor_width / 2, n_pixel)

    @cached_property
    def images(self):
        fits_path = self.path / 'FITS Files'
        images = self.initialize_images()
        for n, tmstp in enumerate(tqdm(self.tmstps, desc='load images', leave=False)):
            file = fits_path / tmstp.strftime(self.strftime + '_full.fts')
            if file.is_file():
                fits_file = FitsFile(file)
                shot = images.shot.sel(tmstp=tmstp)

                for i, image in enumerate(fits_file[0]):
                    images['image_' + str(i).zfill(2)].loc[shot] = image
        return images

    @cached_property
    def scope_traces(self):
        scope_traces = self.initialize_traces()
        for tmstp in tqdm(self.tmstps, desc='load scope traces', leave=False):
            shot = scope_traces.shot.sel(tmstp=tmstp)
            scope_traces.loc[shot] = self.fast_scope_trace(tmstp)
        return scope_traces
    
    @property
    def shot_multi_index(self):
        shot_multi_index = self.variables.reset_index()
        return pd.MultiIndex.from_frame(shot_multi_index)

    def initialize_traces(self):
        for tmstp in self.tmstps:
            trace = self.single_scope_trace[tmstp]
            if trace is not None:
                shape = (len(self.tmstps), trace.size)

                scope_traces = xr.DataArray(
                    np.full(shape, np.NaN, dtype=np.float32),
                    dims=['shot', 'time'],
                    coords={'shot': self.shot_multi_index, 'time': trace.index}
                )
                return scope_traces
        raise AttributeError("No scope_traces are found.")

    def initialize_images(self):
        for tmstp in self.tmstps:
            file = self.path / 'FITS Files' / tmstp.strftime(self.strftime + '_full.fts')
            if file.is_file():
                fits_file = FitsFile(file)
                image = fits_file[0]

                n_images, pixel_y, pixel_x = image.shape
                shape = (len(self.tmstps), pixel_y, pixel_x)
                empty_image = np.full(shape, np.NaN, dtype=np.float32)

                x = self.get_image_coords(image[0], 0)
                y = self.get_image_coords(image[0], 1)

                images = xr.Dataset(
                    {'image_' + str(i).zfill(2): (['shot', 'x', 'y'], empty_image.copy()) for i in range(n_images)},
                    coords={
                        'shot': self.shot_multi_index,
                        'x': x,
                        'y': y}
                )

                return images
        raise AttributeError("No images are found.")

    def save_raw_data(self, path=None):
        if path is None:
            path = self.path
        data = self.raw_data.reset_index('shot')
        data.to_netcdf(path / 'raw_data.h5')

    @cached_property
    def raw_data(self):
        raw_data = xr.Dataset()
        try:
            raw_data = xr.merge([raw_data, self.images])
        except AttributeError:
            pass
        try:
            raw_data['scope_traces'] = self.scope_traces
        except AttributeError:
            pass
        raw_data.attrs.update(self.parameters)
        return raw_data

    @GetterWithTimestamp
    def single_image(self, tmstp):
        path = self['FITS Files'][tmstp.strftime(self.strftime + '_full.fts')].path
        fits_file = FitsFile(path)
        images = fits_file[0]

        images = xr.Dataset(
            {'image_' + str(i).zfill(2): (['x', 'y'], image) for i, image in enumerate(images)},
            coords={
                'x': self.get_image_coords(images[1], 0),
                'y': self.get_image_coords(images[1], 1),
                'tmstp': tmstp
            }
        )
        images = images.assign_coords(self.variables.loc[tmstp])
        images = images.set_index(shot=list(self.variables.columns) + ['tmstp'])
        return images

    @GetterWithTimestamp
    def single_parameters(self, tmstp):
        parameter_path = self.path / 'Variables' / tmstp.strftime(self.strftime + '.txt')
        parameters = pd.read_csv(parameter_path, **self.csv_kwargs)
        #_parameters_class = self['Variables'].lazy_get(tmstp.strftime(self.strftime + '.txt'))
        # parameters = _parameters_class.read(**self.csv_kwargs)
        parameters.name = tmstp
        parameters.index.name = None
        parameters.drop('dummy', inplace=True)
        return parameters

    @GetterWithTimestamp
    def exp_seq(self, tmstp):
        return self['Experimental Sequences'][tmstp.strftime(self.strftime + '.xml')]

    @GetterWithTimestamp
    def single_scope_trace(self, tmstp):
        file = self.path / 'Scope Traces' / tmstp.strftime(self.strftime + '_C1.csv')
        if file.is_file():
            scope_trace = pd.read_csv(file, **self.csv_kwargs)
            if scope_trace.shape[0] > 1:
                scope_trace.name = tmstp
                scope_trace.index.name = 'time'
                return scope_trace

    def fast_scope_trace(self, tmstp):
        file = self.path / 'Scope Traces' / tmstp.strftime(self.strftime + '_C1.csv')
        if file.is_file():
            scope_trace = pd.read_csv(file, **self.fast_csv_kwargs)
            if scope_trace.shape[0] > 1:
                return scope_trace.values

    @GetterWithTimestamp
    def voltage(self, tmstp):
        return self['Voltages'][tmstp.strftime('voltages_' + self.strftime + '.xml')]

    def single_shot_from_tmstp(self, path, tmstp):
        tmstp_str = tmstp.strftime(ExpSequence.TIME_FORMAT)
        file_path = join(path, tmstp_str + '.h5')
        single_shot = xr.Dataset()
        single_shot.attrs.update(self.single_parameters[tmstp])
        try:
            single_shot.update(self.single_image[tmstp])
        except KeyError:
            pass
        try:
            single_shot['single_scope_trace'] = self.single_scope_trace[tmstp]
        except KeyError:
            pass
        single_shot.x.attrs.update(units='um')
        single_shot.y.attrs.update(units='um')
        single_shot.time.attrs.update(units='milliseconds')
        single_shot.to_netcdf(file_path)

    def create_new(self, path):
        UserWarning(" Don't us e this method, it is super slow. Rather use old_seq.save_raw_data()")

        if isdir(path):
            if click.confirm('Sequence already exists. Do you want to delete the old sequence and create a new?',
                             default=True):
                shutil.rmtree(path)
            else:
                print('Did not create new sequence')
                return Directory(path)
        exp_seq = Directory(path)
        exp_seq['Experimental Sequences'] = self['Experimental Sequences']
        exp_seq['Voltages'] = self['Voltages']
        try:
            self.old_la.to_csv(join(exp_seq.path, 'old_la.csv'))
        except KeyError:
            pass
        raw_data = Directory(join(path, 'raw_data'))
        for tmstp in tqdm(self.tmstps):
            file_path = tmstp.strftime(self.strftime) + '.h5'
            if file_path in raw_data:
                del exp_seq[file_path]
            self.single_shot_from_tmstp(raw_data.path, tmstp)
        return Directory(path)

    @property
    def old_la(self):
        old_la = self['FITS Files']['00_all_fit_results.csv']
        old_la.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        old_la.time = old_la.time.apply(lambda t: pd.to_datetime(t, unit='s', origin=self.tmstps[0].date()))
        old_la.set_index('time', inplace=True)
        del old_la['Index']
        return old_la

    def get_old_la_from_tmstp(self, tmstp):
        time = tmstp.hour * 60*60 + tmstp.minute * 60 + tmstp.second
        old_la = self.old_la
        old_la = old_la.loc[time]
        old_la.name = tmstp
        return old_la
