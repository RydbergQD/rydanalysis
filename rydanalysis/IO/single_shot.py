from rydanalysis.IO.directory import Directory

import pandas as pd


class SingleShot(Directory):
    """
    Analysis of a single experimental run.

    path: folder location of the run.

    Folder has the following structure:
        yyyy_mm_dd_HH.MM.SS:
        - analysis:
          - old_la.csv
          ...
        - exp_data:
          - exp_seq.xml
          - image.fits
          - scope_trace.csv
          - variable.txt
          - voltage.xml
          ...

    """
    def __init__(self, path):
        super(SingleShot, self).__init__(path)

    @property
    def old_la(self):
        return pd.read_csv(self['analysis']['old_la.csv'].path, index_col=0, squeeze=True)

    @property
    def image(self):
        pass

    @property
    def scope_trace(self):
        pass

    @property
    def variable(self):
        pass
