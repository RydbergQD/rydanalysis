from rydanalysis.IO.os import Directory
from rydanalysis.IO.io import GetterWithTimestamp, _load_path
from rydanalysis.IO.exp_sequence import ExpSequence
from rydanalysis.IO.fits import FitsFile

import pandas as pd
from os.path import basename, join, isdir
from tqdm import tqdm_notebook as tqdm
import warnings
import xarray as xr
import numpy as np
import click
import shutil


class OldStructure(Directory):
    strftime = '%Y_%m_%d_%H.%M.%S'
    csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)

    def __init__(self, path, handle_key_errors='ignore', sensor_widths=(428, 2191.36)):
        super(OldStructure, self).__init__(path)
        self.handle_key_errors = handle_key_errors
        self.tmstps = self.get_tmstps()
        self.sensor_widths = sensor_widths
        self.variables, _ = self._find_variables()

    def _find_variables(self):
        _parameters = pd.concat([self.parameters[tmstp] for tmstp in self.tmstps], axis=1)
        variables = _parameters[_parameters.T.nunique() > 1].T
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

    @GetterWithTimestamp
    def images(self, tmstp):
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
        return images

    @GetterWithTimestamp
    def parameters(self, tmstp):
        _parameters_class = self['Variables'].lazy_get(tmstp.strftime(self.strftime + '.txt'))
        parameters = _parameters_class.read(**self.csv_kwargs)
        parameters.name = tmstp
        parameters.index.name = None
        parameters.drop('dummy', inplace=True)
        return parameters

    @GetterWithTimestamp
    def exp_seq(self, tmstp):
        return self['Experimental Sequences'][tmstp.strftime(self.strftime + '.xml')]

    @GetterWithTimestamp
    def scope_trace(self, tmstp):
        path = join(self['Scope Traces'].path, tmstp.strftime(self.strftime + '_C1.csv'))
        scope_trace = pd.read_csv(path, **self.csv_kwargs)
        scope_trace.name = tmstp
        scope_trace.index.name = 'time'
        return scope_trace

    @GetterWithTimestamp
    def voltage(self, tmstp):
        return self['Voltages'][tmstp.strftime('voltages_' + self.strftime + '.xml')]

    def single_shot_from_tmstp(self, path, tmstp):
        tmstp_str = tmstp.strftime(ExpSequence.TIME_FORMAT)
        file_path = join(path, tmstp_str + '.h5')
        single_shot = xr.Dataset()
        single_shot.attrs.update(self.parameters[tmstp])
        try:
            single_shot.update(self.images[tmstp])
        except KeyError:
            pass
        try:
            single_shot['scope_trace'] = self.scope_trace[tmstp]
        except KeyError:
            pass
        single_shot.x.attrs.update(units='µm')
        single_shot.y.attrs.update(units='µm')
        single_shot.time.attrs.update(units='ms')
        single_shot.to_netcdf(file_path)

    def create_new(self, path):
        if isdir(path):
            if click.confirm('Sequence already exists. Do you want to delete the old sequence and create a new?',
                             default=True):
                shutil.rmtree(path)
            else:
                raise FileExistsError('Could not create new Sequence.')
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
