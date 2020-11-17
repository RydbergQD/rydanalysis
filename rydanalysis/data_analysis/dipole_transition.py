import arc
import numpy as np
from scipy.constants import c, h, e, epsilon_0, hbar
from scipy.constants import physical_constants
import xarray as xr

from rydanalysis.auxiliary.decorators import cached_property

a0 = physical_constants["Bohr radius"][0]


class DipoleTransition:
    def __init__(
        self,
        n1=5,
        l1=0,
        j1=0.5,
        n2=5,
        l2=1,
        j2=1.5,
        mj1=0.5,
        mj2=1.5,
        q=1,
        temperature=300,
    ):
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
        return self.atom.getTransitionWavelength(
            self.n1, self.l1, self.j1, self.n2, self.l2, self.j2
        )

    @property
    def frequency(self):
        return c / self.wavelength

    @property
    def dipole_matrix_element(self):
        return (
            self.atom.getDipoleMatrixElement(
                self.n1,
                self.l1,
                self.j1,
                self.mj1,
                self.n2,
                self.l2,
                self.j2,
                self.mj2,
                self.q,
            )
            * e
            * a0
        )

    def get_rabi_freq(self, waist, power):
        return self.atom.getRabiFrequency(
            self.n1,
            self.l1,
            self.j1,
            self.mj1,
            self.n2,
            self.l2,
            self.j2,
            self.q,
            power,
            waist,
        )

    @property
    def life_time(self):
        return self.atom.getStateLifetime(
            self.n2, self.l2, self.j2, self.temperature, self.n2 + 10
        )

    @property
    def decay_rate(self):
        return 1 / self.life_time

    @property
    def cross_section(self):
        return 3 / (2 * np.pi) * self.wavelength ** 2

    def power(self, image, quantum_efficiency, t_exp):
        return h * self.frequency * image / (quantum_efficiency * t_exp)


class LiveAnalysisTransition(DipoleTransition):
    wavelength = 7.802415056628226e-07
    frequency = 384230338714579.7
    dipole_matrix_element = 2.5248537096860405e-29
    life_time = 2.6434501256816698e-08

    def __init__(self):
        super(LiveAnalysisTransition, self).__init__(
            n1=5,
            l1=0,
            j1=0.5,
            n2=5,
            l2=1,
            j2=1.5,
            mj1=0.5,
            mj2=1.5,
            q=1,
            temperature=300,
        )


# class DepletionImaging(ReferenceAnalysis):
#     def __init__(self, with_rydberg_images, no_rydberg_images, only_light_images, background=0, mask=None,
#                  transition_kwargs=None, pca_kwargs=None, binning=2, t_exp=None,
#                  saturation_calc_method='flat_imaging'):
#         super().__init__(no_rydberg_images, background, mask, transition_kwargs,
#                          pca_kwargs, binning, t_exp, saturation_calc_method)
#         self.only_light_images = only_light_images - background
#         self.with_rydberg_images = with_rydberg_images - background
#
#     @property
#     def power(self):
#         self.check_t_exp()
#         light = self.extract_light(self.only_light_images)
#         return h * self.frequency * light / (self.QUANTUM_EFFICIENCY * self.t_exp)
#
#     @property
#     def transmission(self):
#         reference_images = self.optimized_reference_images(self.with_rydberg_images)
#         transmission = self.with_rydberg_images / reference_images
#         return transmission
#
#     @property
#     def optical_depth(self):
#         return -np.log(self.transmission)
#
#     @property
#     def density(self):
#         return (1 + self.saturation_parameter) / self.cross_section * self.optical_depth
