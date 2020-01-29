import arc
import numpy as np
from scipy.constants import c, h, e, epsilon_0, hbar
from scipy.constants import physical_constants
import xarray as xr

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

    def __init__(self, reference_image, background_image, t_exp,
                 n1=5, l1=0, j1=0.5, n2=5, l2=1, j2=1.5, mj1=0.5, mj2=1.5, q=1, binning=2):
        super().__init__(n1=n1, l1=l1, j1=j1, n2=n2, l2=l2, j2=j2, mj1=mj1, mj2=mj2, q=q)
        self.reference_image = reference_image
        self.background_image = background_image
        self.t_exp = t_exp
        self.binning = binning

    @classmethod
    def from_raw_data(cls, raw_data: xr.Dataset):
        t_exp = raw_data.tCAM * 1e-3
        binning = 100/raw_data.x.size
        reference_image = raw_data.image_03.mean('tmstp')
        background_image = raw_data.image_05.mean('tmstp')
        return cls(reference_image, background_image, t_exp, binning=binning)

    def __call__(self, ground_state_image, reference_image, **kwargs):
        transmission = ground_state_image / reference_image
        return - (1 + self.saturation_parameter) / self.cross_section * np.log(transmission)

    @property
    def pixel_size(self):
        return self.binning * self.PIXEL_SIZE

    @property
    def power(self):
        return h * self.frequency * self.reference_image / (self.QUANTUM_EFFICIENCY * self.t_exp)

    @property
    def intensity(self):
        return self.power / self.pixel_size**2

    @property
    def field_strength(self):
        return np.sqrt((2*self.intensity)/(c*epsilon_0))

    @property
    def rabi_frequency(self):
        return self.dipole_matrix_element * self.field_strength / hbar

    @property
    def saturation_parameter(self):
        return self.get_saturation_parameter(self.rabi_frequency)
