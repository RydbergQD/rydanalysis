from rydanalysis.IO.directory import Directory

import pandas as pd
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
        new = Directory(join(path, tmstp.strftime(strftime)))
        new['variable.txt'] = self.get_variable(tmstp)
        new['exp_seq.xml'] = self.get_exp_seq(tmstp)
        new['image.fits'] = self.get_fits(tmstp)
        new['scope_trace.csv'] = self.get_scope_trace(tmstp)
        new['voltage.xml'] = self.get_voltage(tmstp)
        return new

    def create_new(self, path):
        new = Directory(path)
        for tmstp in tqdm(self.tmstps):
            if tmstp.strftime(strftime) in new:
                del new[tmstp.strftime(strftime)]
            self.create_new_from_tmstp(path, tmstp)
        return new
