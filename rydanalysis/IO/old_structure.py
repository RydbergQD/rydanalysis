from rydanalysis.IO.directory import Directory

import pandas as pd
import os
from os.path import basename, join
from tqdm import tqdm_notebook as tqdm


strftime = '%Y_%m_%d_%H.%M.%S'


class OldStructure(Directory):

    def __init__(self, path):
        super(OldStructure, self).__init__(path)

    @property
    def tmstps(self):
        tmstps = []
        for path in self['Variables'].iter_files():
            tmstps.append(pd.to_datetime(basename(path), format=strftime + '.txt'))
        return tmstps

    def get_variable(self, tmstp):
        return self['Variables'][tmstp.strftime(strftime + '.txt')]

    def get_exp_seq(self, tmstp):
        return self['Experimental Sequences'][tmstp.strftime(strftime + '.xml')]

    def get_fits(self, tmstp):
        return self['FITS Files'][tmstp.strftime(strftime + '_full.fts')]

    def get_scope_trace(self, tmstp):
        return self['Scope Traces'][tmstp.strftime(strftime + '_C1.csv')]

    def get_voltage(self, tmstp):
        return self['Voltages'][tmstp.strftime('voltages_' + strftime + '.xml')]

    def create_new_from_tmstp(self, path, tmstp):
        new = Directory(join(path, tmstp.strftime(strftime), 'exp_data'))
        new['variable.txt'] = self.get_variable(tmstp)
        new['exp_seq.xml'] = self.get_exp_seq(tmstp)
        new['image.fits'] = self.get_fits(tmstp)
        new['scope_trace.csv'] = self.get_scope_trace(tmstp)
        new['voltage.xml'] = self.get_voltage(tmstp)
        old_la = self.get_old_la_from_tmstp(tmstp)
        dir_analysis = join(path, tmstp.strftime(strftime), 'analysis')
        os.makedirs(dir_analysis)
        old_la.to_csv(join(dir_analysis, 'old_la.csv'))
        return new

    def create_new(self, path):
        new = Directory(path)
        for tmstp in tqdm(self.tmstps):
            if tmstp.strftime(strftime) in new:
                del new[tmstp.strftime(strftime)]
            self.create_new_from_tmstp(path, tmstp)
        return new

    @property
    def old_la(self):
        file = self['FITS Files']['00_all_fit_results.csv']
        old_la = pd.read_csv(file.path, index_col=0)
        old_la.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        old_la.set_index('time', inplace=True)
        del old_la['Index']
        return old_la

    def get_old_la_from_tmstp(self, tmstp):
        time = tmstp.hour * 60*60 + tmstp.minute * 60 + tmstp.second
        old_la = self.old_la
        return old_la.loc[time]
