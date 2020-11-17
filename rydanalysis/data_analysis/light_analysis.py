import numpy as np
from scipy.constants import c, h, epsilon_0, hbar
import xarray as xr
from typing import Optional

from rydanalysis.data_analysis.dipole_transition import DipoleTransition


class LightAnalysis:
    QUANTUM_EFFICIENCY = 0.44
    SINGLE_PIXEL_SIZE = 2.09e-6

    def __init__(
        self,
        light_images,
        t_exp=Optional[float],
        transition=DipoleTransition(),
        binning=2,
    ):
        """

        Args:
            light_images: xr.DataArray of images, used to calculated saturation parameter
            t_exp: Exposure time of the camera.
            transition: Quantum numbers defining the transition
            binning: binning of the camera, default is a binning=2 (2x2 binning)
        """
        self.light_images = light_images
        self.t_exp = t_exp
        self.transition = transition

        self.binning = binning

    @classmethod
    def from_raw_data(
        cls, raw_data: xr.Dataset, light_mask=None, transition=DipoleTransition()
    ):
        t_exp = raw_data.tCAM * 1e-3
        binning = 100 / raw_data.x.size

        background = raw_data.image_05.mean("shot")
        images = raw_data.image_03 - background
        images = images.where(light_mask)

        return cls(images, t_exp=t_exp, binning=binning, transition=transition)

    @property
    def pixel_size(self):
        return self.binning * self.SINGLE_PIXEL_SIZE

    @property
    def power(self):
        return get_power(
            self.light_images, self.t_exp, self.QUANTUM_EFFICIENCY, self.transition
        )

    @property
    def intensity(self):
        return get_intensity(
            self.light_images,
            self.t_exp,
            self.QUANTUM_EFFICIENCY,
            self.transition,
            self.pixel_size,
        )

    @property
    def field_strength(self):
        return np.sqrt((2 * self.intensity) / (c * epsilon_0))

    @property
    def rabi_frequency(self):
        dipole_matrix_element = self.transition.dipole_matrix_element
        return dipole_matrix_element * self.field_strength / hbar

    @property
    def saturation_parameter(self):
        if self.t_exp is None:
            return 0
        decay_rate = self.transition.decay_rate
        return 2 * (self.rabi_frequency / decay_rate) ** 2

    @property
    def average_saturation(self):
        light = self.light_images.mean(["x", "y"])
        return get_saturation_parameter(
            light, self.t_exp, self.QUANTUM_EFFICIENCY, self.transition, self.pixel_size
        )


def get_power(
    images,
    t_exp,
    quantum_efficiency=LightAnalysis.QUANTUM_EFFICIENCY,
    transition=DipoleTransition(),
):
    frequency = transition.frequency
    return h * frequency * images / (quantum_efficiency * t_exp)


def get_intensity(
    images,
    t_exp,
    quantum_efficiency=LightAnalysis.QUANTUM_EFFICIENCY,
    transition=DipoleTransition(),
    pixel_size=LightAnalysis.SINGLE_PIXEL_SIZE,
):
    power = get_power(images, t_exp, quantum_efficiency, transition)
    return power / pixel_size ** 2


def get_field_strength(
    images,
    t_exp,
    quantum_efficiency=LightAnalysis.QUANTUM_EFFICIENCY,
    transition=DipoleTransition(),
    pixel_size=LightAnalysis.SINGLE_PIXEL_SIZE,
):
    intensity = get_intensity(images, t_exp, quantum_efficiency, transition, pixel_size)
    return np.sqrt((2 * intensity) / (c * epsilon_0))


def get_rabi_frequency(
    images,
    t_exp,
    quantum_efficiency=LightAnalysis.QUANTUM_EFFICIENCY,
    transition=DipoleTransition(),
    pixel_size=LightAnalysis.SINGLE_PIXEL_SIZE,
):
    field_strength = get_field_strength(
        images, t_exp, quantum_efficiency, transition, pixel_size
    )
    dipole_matrix_element = transition.dipole_matrix_element
    return dipole_matrix_element * field_strength / hbar


def get_saturation_parameter(
    images,
    t_exp=Optional[float],
    quantum_efficiency=LightAnalysis.QUANTUM_EFFICIENCY,
    transition=DipoleTransition(),
    pixel_size=LightAnalysis.SINGLE_PIXEL_SIZE,
):
    if t_exp is None:
        return 0
    decay_rate = transition.decay_rate
    rabi_frequency = get_rabi_frequency(
        images, t_exp, quantum_efficiency, transition, pixel_size
    )
    return 2 * (rabi_frequency / decay_rate) ** 2
