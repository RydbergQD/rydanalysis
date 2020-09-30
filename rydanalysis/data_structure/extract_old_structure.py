from prefect import task, Flow, Parameter, unmapped
import pandas as pd
import astropy.io.fits
import numpy as np
import xarray as xr
from pathlib import Path
from dataclasses import dataclass, field
from rydanalysis.auxiliary.prefect_aux import PrefectParams
from typing import List


class OldStructureN:
    raw_data: xr.Dataset
    strftime = '%Y_%m_%d_%H.%M.%S'
    date_strftime = '%Y_%m_%d'
    filename_pattern = '????_??_??_??.??.??'
    csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
    fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)

    def __init__(self, path, handle_key_errors='ignore', sensor_widths=(1100, 214),
                 initial_update=True):
        self.path = Path(path)

    @classmethod
    def build_flow(cls):
        with Flow("extract old structure") as flow:
            # load data and path
            data = Parameter("data")
            path = Parameter("path")

            # load parameters
            strftime = Parameter("strftime", default='%Y_%m_%d_%H.%M.%S')
            date_strftime = Parameter("data_strftime", '%Y_%m_%d')
            filename_pattern = Parameter("filename_pattern", '????_??_??_??.??.??')
            csv_kwargs = dict(index_col=0, squeeze=True, sep='\t', decimal=',', header=None)
            fast_csv_kwargs = dict(usecols=[1], squeeze=True, sep='\t', decimal=',', header=None)

            # Extraction
            tmstps = extract_tmstps(data, path, strftime, filename_pattern)

            parameters = extract_parameters.map(tmstps)
            constant_parameters = reduce_parameters(parameters)

            images = extract_image.map(tmstps, unmapped(path), unmapped(strftime))
        return flow


@dataclass
class FastCSVParams(PrefectParams):
    name = "csv params"
    index_col: int = 0
    squeeze: bool = True
    sep: str = '\t'
    decimal: str = ','
    header: any = None


@dataclass
class CSVParams(PrefectParams):
    name = "csv params"
    usecols: List[int] = field(default=[1])
    squeeze: bool = True
    sep: str = '\t'
    decimal: str = ','
    header: any = None


def iter_tmstps(path: Path, strftime: str, filename_pattern: str):
    for sub_path in path.glob(filename_pattern):
        try:
            tmstp = pd.to_datetime(sub_path.name, format=strftime + '.txt')
            yield tmstp
        except ValueError:
            print("couldn't read {0}. Skipping this file...".format(sub_path.name))


@task
def extract_tmstps(data: xr.Dataset, path: Path, strftime: str, filename_pattern: str):
    variables_path = path / 'Variables'
    new_tmstps = [tmstp for tmstp in
                  iter_tmstps(variables_path, strftime, filename_pattern + '.txt')
                  if tmstp not in data.tmstps.values]
    return new_tmstps


@task
def extract_parameters(tmstp, path, strftime, csv_kwargs):
    parameter_path = path / 'Variables' / tmstp.strftime(strftime + '.txt')
    parameters = pd.read_csv(parameter_path, **csv_kwargs)
    parameters.name = tmstp
    parameters.index.name = None
    parameters.drop('dummy', inplace=True)
    return parameters


@task
def reduce_parameters(parameters):
    _parameters = pd.concat(parameters, axis=1)
    _parameters = _parameters.T
    _parameters.index.name = 'tmstp'
    return _parameters


@task
def extract_image(tmstp, path, strftime='%Y_%m_%d_%H.%M.%S'):
    fits_path = path / 'FITS Files'
    file = fits_path / tmstp.strftime(strftime + '_full.fts')
    if file.is_file():
        with astropy.io.fits.open(file) as fits_file:
            data = fits_file[0].data
            return np.transpose(data, axes=[0, 2, 1])


def _get_image_coords(image):
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

@task
def reduce_images(images):
    xr.concat(images, )
