import numpy as np
from scipy.constants import physical_constants
import xarray as xr

from rydanalysis.data_analysis.dipole_transition import LiveAnalysisTransition, DipoleTransition
from rydanalysis.data_analysis.light_analysis import LightAnalysis
from rydanalysis.auxiliary.pca import PCAXarray

a0 = physical_constants['Bohr radius'][0]


class AbsorptionImaging(LightAnalysis):
    def __init__(self, absorption_images: xr.DataArray, light_images: xr.DataArray, t_exp: float,
                 transition=DipoleTransition(), binning: int = 2):
        super(AbsorptionImaging, self).__init__(light_images, t_exp, transition=transition, binning=binning)
        self.absorption_images = absorption_images

    @classmethod
    def from_raw_data(cls, data: xr.Dataset, atoms_mask=None, light_mask=True, transition=DipoleTransition(),
                      pca_kwargs=None, background_name='image_05', light_name='image_03', atom_name='image_01'):
        """
        Method to compute density. Uses pca to find optimal reference images.
        Args:
            data: experimental Data
            atoms_mask: True at the position of the atoms
            light_mask: True where light hits camera
            transition: Dipole transition
            pca_kwargs: keyword arguments for pca analysis
            background_name: Name of background image
            light_name: Name of light image
            atom_name: Name of atom image

        Returns:
            Absorption imaging instance
        """
        if pca_kwargs is None:
            pca_kwargs = dict()
        t_exp = data.tCAM * 1e-3
        binning = 100 / data.x.size
        background = data[background_name].where(light_mask).mean('shot')

        absorption_images = data[atom_name] - background
        absorption_images = absorption_images.where(light_mask)

        light_images = data[light_name] - background
        light_images = light_images.where(light_mask)
        pca: PCAXarray = light_images.pca(**pca_kwargs)
        no_atoms_mask = np.logical_not(atoms_mask)
        light_images = pca.find_references(absorption_images, no_atoms_mask)

        return cls(absorption_images, light_images, t_exp=t_exp, binning=binning, transition=transition)

    @classmethod
    def for_live_analysis(cls, shot, transition=LiveAnalysisTransition(),
                          background_name='image_05', light_name='image_03', atom_name='image_01'):
        """
        Method to compute density. Uses only a single run of the average.
        Args:
            shot: experimental data (single shot)
            transition: Dipole transition
            background_name: Name of background image
            light_name: Name of light image
            atom_name: Name of atom image

        Returns:
            Absorption imaging instance
        """
        t_exp = shot.tCAM * 1e-3
        binning = 100 / shot.x.size
        background = shot[background_name]

        absorption_images = shot[atom_name] - background
        light_images = shot[light_name] - background

        return cls(absorption_images, light_images, t_exp=t_exp, binning=binning, transition=transition)

    @property
    def transmission(self):
        return get_transmission(self.absorption_images, self.light_images)

    @property
    def optical_depth(self):
        return get_optical_depth(self.absorption_images, self.light_images)

    @property
    def density(self) -> xr.DataArray:
        saturation_parameter = self.average_saturation
        return get_density(self.absorption_images, self.light_images, saturation_parameter, self.transition)


def get_transmission(absorption_images, reference_images):
    return absorption_images / reference_images


def get_optical_depth(absorption_images, reference_images):
    transmission = get_transmission(absorption_images, reference_images)
    return -np.log(transmission)


def get_density(absorption_images, reference_images, saturation_parameter, transition=DipoleTransition()):
    cross_section = transition.cross_section
    optical_depth = get_optical_depth(absorption_images, reference_images)
    return (1 + saturation_parameter) / cross_section * optical_depth
