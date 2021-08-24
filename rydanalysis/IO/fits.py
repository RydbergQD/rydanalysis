from os.path import isfile, basename
from collections.abc import MutableSequence
import astropy.io.fits
from yaml import dump

from rydanalysis.IO.io import load_fits, tree


class FitsFile(MutableSequence):

    def __init__(self, path):
        if not isfile(path):
            with astropy.io.fits.open(path, 'ostream') as _:
                pass
        self.path = path
        self.__name__ = basename(path)

    def __getitem__(self, fits_index):
        if fits_index >= len(self):
            raise IndexError
        return load_fits(self.path, fits_index)

    def lazy_get(self, fits_index):
        if fits_index >= len(self):
            raise IndexError
        return load_fits(self.path, fits_index, lazy=True)

    def __setitem__(self, index, data):
        hdu = astropy.io.fits.PrimaryHDU(data)
        with astropy.io.fits.open(self.path, 'update') as f:
            f[index] = hdu

    def insert(self, index, data):
        hdu = astropy.io.fits.PrimaryHDU(data)
        with astropy.io.fits.open(self.path, 'update') as f:
            f.insert(index, hdu)

    def items(self):
        return enumerate(self)

    def __delitem__(self, key):
        with astropy.io.fits.open(self.path, 'update') as f:
            f.remove(f[key])

    def __len__(self):
        with astropy.io.fits.open(self.path) as f:
            return len(f)

    def __repr__(self):
        return "FitsFile: " + self.path

    def __str__(self):
        return "FitsFile: " + self.__name__

    def tree(self, include_files='all'):
        return print(dump(tree(self, include_files)))


class FitsDataset:
    def __init__(self, path, fits_index):
        self.path = path
        self.fits_index = fits_index
        self.__name__ = fits_index

    def __repr__(self):
        with astropy.io.fits.open(self.path) as hf:
            return str(hf[self.fits_index])

    def read(self):
        with astropy.io.fits.open(self.path) as f:
            return f[self.fits_index].data
