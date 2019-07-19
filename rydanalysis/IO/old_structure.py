from rydanalysis.IO.os import Directory
from rydanalysis.IO.exp_sequence import ExpSequence
from rydanalysis.auxiliary.warnings_and_errors import conditional_waning

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
        path = self['Variables'][tmstp.strftime(strftime + '.txt')].path
        variables = pd.read_csv(path, index_col=0, squeeze=True, sep='\t',
                                decimal=',', header=None)
        variables.name = 'variables'
        variables.index.name = 'variable'
        return variables

    def get_exp_seq(self, tmstp):
        return self['Experimental Sequences'][tmstp.strftime(strftime + '.xml')]

    def get_fits(self, tmstp):
        return self['FITS Files'][tmstp.strftime(strftime + '_full.fts')]

    def get_scope_trace(self, tmstp):
        path = self['Scope Traces'][tmstp.strftime(strftime + '_C1.csv')].path
        scope_trace = pd.read_csv(path, index_col=0, sep='\t', header=None,
                                  decimal=',', squeeze=True)
        scope_trace.name = 'MCP'
        scope_trace.index.name = 'time'
        return scope_trace

    def get_voltage(self, tmstp):
        return self['Voltages'][tmstp.strftime('voltages_' + strftime + '.xml')]

    def create_new_from_tmstp(self, path, tmstp, ignore_warnings=True):
        new = Directory(join(path, tmstp.strftime(strftime), 'exp_data'))
        new['exp_seq.xml'] = self.get_exp_seq(tmstp)
        variables = self.get_variable(tmstp)
        variables.to_csv(join(new.path, 'parameters.csv'), header=True)
        new['image.fits'] = self.get_fits(tmstp)
        try:
            scope_trace = self.get_scope_trace(tmstp)
            scope_trace.to_csv(join(new.path, 'scope_trace.csv'), header=True)
<<<<<<< HEAD

        except:
            pass#print("No scope traces present in run with timestamp")
        try:
            new['voltage.xml'] = self.get_voltage(tmstp)
        except:
            pass#print("No voltage readings present in run with timestamp")

=======
        except KeyError:
            conditional_waning("No scope traces present in run with timestamp", ignore_warnings)
        try:
            new['voltage.xml'] = self.get_voltage(tmstp)
        except KeyError:
            conditional_waning("No voltage readings present in run with timestamp", ignore_warnings)
>>>>>>> f31d7dcc071d0afea10ad5e66c32cc1ec9b5dda1
        dir_analysis = join(path, tmstp.strftime(strftime), 'analysis')
        os.makedirs(dir_analysis)
        try:
            old_la = self.get_old_la_from_tmstp(tmstp)
            old_la.to_csv(join(dir_analysis, 'old_la.csv'), header=True)
<<<<<<< HEAD

        except:
            pass#print("couldn't copy old live analysis")

=======
        except KeyError:
            conditional_waning("couldn't copy old live analysis", ignore_warnings)
        return new
>>>>>>> f31d7dcc071d0afea10ad5e66c32cc1ec9b5dda1

    def create_new(self, path):
        new = Directory(path)
        for tmstp in tqdm(self.tmstps):
            if tmstp.strftime(strftime) in new:
                del new[tmstp.strftime(strftime)]
            self.create_new_from_tmstp(path, tmstp)
        return ExpSequence(path)

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
        old_la = old_la.loc[time]
        old_la.name = tmstp
        return old_la
