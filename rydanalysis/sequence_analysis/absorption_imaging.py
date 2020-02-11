import arc
import numpy as np
from scipy.constants import c, h, e, epsilon_0, hbar
from scipy.constants import physical_constants
import xarray as xr

from rydanalysis.auxiliary.decorators import cached_property

a0 = physical_constants['Bohr radius'][0]


class DipoleTransition:
    def __init__(self, n1=5, l1=0, j1=0.5, n2=5, l2=1, j2=1.5, mj1=0.5, mj2=1.5, q=1, temperature=300):
        self.n1 = n1
        self.l1 = l1
        self.j1 = j1
        self.n2 = n2
        self.l2 = l2
        self.j2 = j2
        self.mj1 = mj1
        self.mj2 = mj2
        self.q = q
        self.atom = arc.Rubidium87()
        self.temperature = temperature

    @property
    def wavelength(self):
        return self.atom.getTransitionWavelength(self.n1, self.l1, self.j1, self.n2, self.l2, self.j2)

    @property
    def frequency(self):
        return c / self.wavelength

    @property
    def dipole_matrix_element(self):
        return self.atom.getDipoleMatrixElement(self.n1, self.l1, self.j1, self.mj1,
                                                self.n2, self.l2, self.j2, self.mj2, self.q)*e*a0

    def get_rabi_freq(self, waist, power):
        return self.atom.getRabiFrequency(self.n1, self.l1, self.j1, self.mj1,
                                          self.n2, self.l2, self.j2, self.q,
                                          power, waist)

    @property
    def life_time(self):
        return self.atom.getStateLifetime(self.n2, self.l2, self.j2, self.temperature, self.n2 + 10)

    @property
    def decay_rate(self):
        return 1 / self.life_time

    def get_saturation_parameter(self, rabi_frequency):
        return 2 * (rabi_frequency / self.decay_rate)**2

    @property
    def cross_section(self):
        return 3/(2*np.pi) * self.wavelength


class AbsorptionImaging(DipoleTransition):
    QUANTUM_EFFICIENCY = 0.44
    PIXEL_SIZE = 2.09e-6

    def __init__(self, reference_images, background=0, mask=None,
                 transition_kwargs=None, pca_kwargs=None, binning=2, t_exp=None, saturation_calc_method='flat_imaging'):
        """

        Args:
            reference_images: xr.DataArray of reference images, used to calculated saturation parameter and for pca
            background: background of the camera, this is going to be averaged and subtracted
             from the light and atom images.
            mask: defines the region o the cloud. The edge region is given by np.logical_not(mask).
            transition_kwargs: Quantum numbers defining the transition
            pca_kwargs: kwargs defining the pca, most important is n_components, which defaults to 30.
            binning: binning of the camera, default is a binning=2 (2x2 binning)
            t_exp: Exposure time of the camera. If t_exp=None, the saturation parameter is set to 0. Default is None.
            saturation_calc_method: To calculate the saturation parameter, the fringes need to be removed from the
             reference image. By default, this is done by assuming an imaging beam larger than the cloud, hence the
              saturation parameter is averaged over the light intensity in the masked region.
        """
        if transition_kwargs is None:
            transition_kwargs = {}
        self.transition_kwargs = transition_kwargs
        super().__init__(**transition_kwargs)

        self.background = background
        self.reference = reference_images - background
        self.mask = mask

        self.t_exp = t_exp
        self.saturation_calc_method = saturation_calc_method
        self.binning = binning

        if pca_kwargs is None:
            pca_kwargs = {}
        if 'n_components' not in pca_kwargs.keys():
            pca_kwargs.update(n_components=30)
        self.pca_kwargs = pca_kwargs

    @classmethod
    def from_raw_data(cls, raw_data: xr.Dataset, mask=None, transition_kwargs=None, pca_kwargs=None):
        t_exp = raw_data.tCAM * 1e-3
        binning = 100/raw_data.x.size

        reference_image = raw_data.image_03
        background = raw_data.image_05.mean('shot')

        return cls(reference_image, background=background, t_exp=t_exp, binning=binning, mask=mask,
                   transition_kwargs=transition_kwargs, pca_kwargs=pca_kwargs)

    @cached_property
    def pca(self):
        return self.reference.pca(**self.pca_kwargs)

    def optimized_reference_images(self, images):
        """
        Build reference images from pca.
        """
        edge_mask = np.logical_not(self.mask)
        return self.pca.find_references(images.where(edge_mask))

    def calculate_density(self, images, remove_background=True):
        if remove_background:
            images = images - self.background
        reference_images = self.optimized_reference_images(images)
        transmission = images / reference_images
        return - (1 + self.saturation_parameter) / self.cross_section * np.log(transmission)

    @property
    def pixel_size(self):
        return self.binning * self.PIXEL_SIZE

    def extract_light(self, light_images):
        if self.saturation_calc_method == 'flat_imaging':
            return light_images.where(self.mask).mean()
        else:
            raise NotImplementedError("The method {} is not yet implemented".format(self.saturation_calc_method))

    @property
    def power(self):
        self.check_t_exp()
        light = self.extract_light(self.reference)
        return h * self.frequency * light / (self.QUANTUM_EFFICIENCY * self.t_exp)

    @property
    def intensity(self):
        self.check_t_exp()
        light = self.extract_light()
        return self.power / self.pixel_size**2

    @property
    def field_strength(self):
        self.check_t_exp()
        return np.sqrt((2*self.intensity)/(c*epsilon_0))

    @property
    def rabi_frequency(self):
        self.check_t_exp()
        return self.dipole_matrix_element * self.field_strength / hbar

    @property
    def saturation_parameter(self):
        if self.t_exp is None:
            return 0
        return self.get_saturation_parameter(self.rabi_frequency)

    def check_t_exp(self):
        if self.t_exp is None:
            raise ValueError("Not available if no value for tEXP is given")


class DepletionImaging(AbsorptionImaging):
    def __init__(self, no_rydberg_images, only_light_images, background=0, mask=None,
                 transition_kwargs=None, pca_kwargs=None, binning=2, t_exp=None,
                 saturation_calc_method='flat_imaging'):
        super().__init__(no_rydberg_images, background, mask, transition_kwargs,
                         pca_kwargs, binning, t_exp, saturation_calc_method)
        self.only_light_images = only_light_images

    @property
    def power(self):
        self.check_t_exp()
        light = self.extract_light(self.only_light_images)
        return h * self.frequency * light / (self.QUANTUM_EFFICIENCY * self.t_exp)
