from os.path import basename
import pandas as pd
import xarray as xr

class File:
    def __init__(self, path):
        self.path = path
        self.__name__ = basename(path)

    def read(self, *args, **kwargs):
        raise NotImplementedError("File type is not yet implemented")

    def __repr__(self):
        return "File: " + self.path

    def __str__(self):
        return "file: " + self.__name__

    @property
    def file_type(self):
        file_name = basename(self.path)
        return file_name.split('.')[-1]


class CSVFile(File):
    csv_kwargs = dict(index_col=0, squeeze=True)

    def __init__(self, path):
        super().__init__(path)

    def read(self, *args, **kwargs):
        csv_kwargs = self.csv_kwargs.copy()
        csv_kwargs.update(**kwargs)
        return pd.read_csv(self.path, *args, **kwargs)

    def __repr__(self):
        return "CSVFile: " + self.path

    def __str__(self):
        return "CSVFile: " + self.__name__
    
class NetCDFFile(File):
    def __init__(self, path):
         super().__init__(path)
         
    def read(self, *args, **kwargs):
        return xr.open_dataset(self.path)

    def __repr__(self):
        return "CSVFile: " + self.path

    def __str__(self):
        return "CSVFile: " + self.__name__
