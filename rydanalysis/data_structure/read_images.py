from typing import Tuple, List
import pandas as pd
from pathlib import Path
import astropy.io.fits
import numpy as np
import xarray as xr
from rydanalysis.auxiliary.user_input import custom_tqdm


# Read images


def read_image(tmstp: pd.Timestamp, path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S"):
    fits_path = path / "FITS Files"
    file = fits_path / tmstp.strftime(strftime + "_full.fts")
    if file.is_file():
        with astropy.io.fits.open(file) as fits_file:
            data = fits_file[0].data
            return np.transpose(data, axes=[0, 2, 1])


def read_image_coords(image, sensor_widths: Tuple[int, int] = (1100, 214)):
    n_images, n_pixel_x, n_pixel_y = image.shape
    sensor_width_x, sensor_width_y = sensor_widths
    sensor_width_x *= 1 - 1 / n_pixel_x
    sensor_width_y *= 1 - 1 / n_pixel_y

    x = np.linspace(-sensor_width_x / 2, sensor_width_x / 2, n_pixel_x)
    y = np.linspace(-sensor_width_y / 2, sensor_width_y / 2, n_pixel_y)
    image_names = ["image_" + str(i).zfill(2) for i in range(n_images)]

    return image_names, x, y


def find_first_image(
    tmstps: List[pd.Timestamp], path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S"
):
    for tmstp in tmstps:
        image = read_image(tmstp, path, strftime)
        return image


def initialize_images(
    tmstps: List[pd.Timestamp], path: Path, strftime: str = "%Y_%m_%d_%H.%M.%S"
):
    try:
        image = find_first_image(tmstps, path, strftime)
        image_names, x, y = read_image_coords(image)
    except AttributeError:
        return None
    shape = (len(tmstps), len(x), len(y))
    empty_image = np.full(shape, np.NaN, dtype=np.float32)

    images = xr.Dataset(
        {name: (["tmstp", "x", "y"], empty_image.copy()) for name in image_names},
        coords={"tmstp": tmstps, "x": x, "y": y},
    )
    return images


def read_images(
    tmstps: List[pd.Timestamp],
    path: Path,
    strftime: str = "%Y_%m_%d_%H.%M.%S",
    interface: str = "tqdm",
):
    images = initialize_images(tmstps, path, strftime)
    if not images:
        return None
    for tmstp in custom_tqdm(tmstps, interface, "Read images...", leave=True):
        image_list = read_image(tmstp, path, strftime)
        if image_list is not None:
            for i, image in enumerate(image_list):
                name = "image_" + str(i).zfill(2)
                images[name].loc[{"tmstp": tmstp}] = image
    return images
