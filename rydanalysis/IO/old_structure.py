from rydanalysis.IO.os import Directory
from rydanalysis.IO.io import GetterWithTimestamp, _load_path
from rydanalysis.IO.h5 import h5_join
from rydanalysis.IO.single_shot import SingleShot
from rydanalysis.IO.fits import FitsFile

import pandas as pd
from os.path import basename, join
from tqdm import tqdm_notebook as tqdm
import warnings


class OldStructure(Directory):
    strftime = '%Y_%m_%d_%H.%M.%S'
    csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)

    def __init__(self, path, handle_key_errors='ignore'):
        super(OldStructure, self).__init__(path)
        self.handle_key_errors = handle_key_errors
        self.tmstps = self.get_tmstps()

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
            tmstps.append(pd.to_datetime(basename(path), format=self.strftime + '.txt'))
        return tmstps

    @GetterWithTimestamp
    def images(self, tmstp):
        path = self['FITS Files'][tmstp.strftime(self.strftime + '_full.fts')].path
        fits_file = FitsFile(path)
        return fits_file[0]

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
        single_shot = SingleShot.initiate_new(path, tmstp)
        single_shot['parameters'] = self.parameters[tmstp]
        for i, image in enumerate(self.images[tmstp]):
            single_shot[h5_join('images', 'image_' + str(i).zfill(2))] = image
        try: single_shot['scope_trace'] = self.scope_trace[tmstp]
        except: pass
        return single_shot

    def create_new(self, path):
        exp_seq = Directory(path)
        exp_seq['Experimental Sequences'] = self['Experimental Sequences']
        exp_seq['Voltages'] = self['Voltages']
        self.old_la.to_csv(join(exp_seq.path, 'old_la.csv'))
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
