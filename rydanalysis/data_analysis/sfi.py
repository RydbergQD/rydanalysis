from distributed.worker import weight
from rydanalysis.fitting.fitting_1d.fitting1d import CosineModel, fit_results_to_dict
from lmfit import Model, Parameters
from functools import cached_property
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt


class MagnCalculator:
    data_color = "tab:blue"
    state_0_color = "tab:orange"
    state_1_color = "tab:green"

    def __init__(self, peak_0, peak_1, bw_method=5e-2, bins=np.linspace(0, 100, 101),
     rydberg_states=["56S", "57S"]):
        self.peak_0 = peak_0
        self.peak_1 = peak_1
        self.bw_method = bw_method
        self.bins=bins
        self.states = rydberg_states

        self.kde_48 = self.gaussian_kde(peak_0)
        self.kde_49 = self.gaussian_kde(peak_1)

    @classmethod
    def from_peak_df(cls, peak_df, bw_method=5e-2, bins=np.linspace(0, 100, 101),
     rydberg_states=["56S", "57S"]):
        peak_0 = peak_df.xs([0, 0], level=["InitON", "Roff"])
        peak_1 = peak_df.xs([2, 0], level=["InitON", "Roff"])
        return cls(peak_0.peak_time, peak_1.peak_time, bw_method, bins, rydberg_states)

    def gaussian_kde(self, peak):
        return gaussian_kde(peak, self.bw_method)

    def model_func(self, x, magn):
        a = (2*magn + 1)/2
        b = 1 - a
        return a * self.kde_49(x) + b*self.kde_48(x)

    @cached_property
    def model(self):
        return Model(self.model_func)

    @cached_property
    def params(self):
        params = Parameters()
        params.add("magn", value=0, min=-0.7, max=0.7)
        return params

    def plot_calibration(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.bins, self.kde_48(self.bins), color=self.state_0_color)
        ax.hist(self.peak_0, bins=self.bins, alpha=0.7, label=self.states[0], density=True, color=self.state_0_color)
        ax.plot(self.bins, self.kde_49(self.bins), color=self.state_1_color)
        ax.hist(self.peak_1, bins=self.bins, alpha=0.7, label=self.states[1], density=True, color=self.state_1_color)
        ax.set_xlabel(r"arrival time [$\mu$s]")
        ax.set_ylabel("peak density [1/$\mu$s]")
        ax.legend()
        return ax

    def fit_result(self, peaks):
        kde = self.gaussian_kde(peaks)
        result = self.model.fit(kde(self.bins), x=self.bins, params=self.params, method="powell")
        result = self.model.fit(kde(self.bins), x=self.bins, params=result.params)
        return result

    def get_magn(self, peaks, to_dict=True):
        result = self.fit_result(peaks)
        if to_dict:
            result = fit_results_to_dict(result)
        return result

    def plot_result(self, peaks, ax=None):
        bins = self.bins
        result = self.get_magn(peaks, to_dict=False)
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(peaks, bins=bins, density=True, label="data", color=self.data_color, alpha=0.7)
        ax.plot(bins, result.eval(x=bins), label="best fit", color=self.data_color)
        magn = result.best_values["magn"]
        a = (2*magn + 1)/2
        ax.plot(bins, a*self.kde_49(bins), label=self.states[1], color=self.state_1_color)
        ax.plot(bins, (1 - a)*self.kde_48(bins), label=self.states[0], color=self.state_0_color)
        print(f"magnetization = {magn}")
        ax.text(0.8, 0.8, f"magnetization = {magn}")
        ax.set_xlabel(r"arrival time [$\mu$s]")
        ax.set_ylabel("peak density [1/$\mu$s]")
        ax.legend()
        return ax

    def get_phase_fit(self, peak_df, model=CosineModel(), params=None, to_dict=True, freq=1/360):
        if params is None:
            params = model.make_params()
            params.add("freq", value=freq, vary=False)
        phase_scan = peak_df.peak_time.groupby("PhaseShift").apply(self.get_magn)
        phase_scan = phase_scan.unstack()
        phase_fit = model.fit(phase_scan["magn"], x=phase_scan.index, params=params,
         weights=1/phase_scan["magn_stderr"])
        if to_dict:
            phase_fit = fit_results_to_dict(phase_fit)
        return phase_fit

    def plot_phase_fit(self, peak_df, model=CosineModel(), params=None, ax=None):
        phase_fit = self.get_phase_fit(peak_df, model, params, to_dict=False)
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(phase_fit.data.index, phase_fit.data, 1/phase_fit.weights, linestyle='', marker='o')
        phases = np.linspace(0, 1/phase_fit.params["freq"].value, 51)
        ax.plot(phases, phase_fit.eval(x=phases))
        ax.set_xlabel("phase [°]")
        ax.set_ylabel("magnetization")
        return ax

    def get_amp_df(self, peak_df):
        names = self.extract_names(peak_df)
        amp_df = peak_df.groupby(names).progress_apply(self.get_phase_fit)
        return amp_df

    def extract_names(self, peak_df):
        names = [name for name in peak_df.index.names
         if name not in ["peak_number", "tmstp", "PhaseShift"]]
        return names

    def get_z_magn_df(self, peak_df):
        names = self.extract_names(peak_df)
        z_magn = peak_df.peak_time.groupby(names).progress_apply(self.get_magn)
        return z_magn.unstack()

    def get_magn_df_ramsey(self, peak_df):
        peak_xy = peak_df.xs([1, 1], level=["InitON", "Roff"], drop_level=True)
        magn_df = self.get_amp_df(peak_xy)
        peak_z = peak_df.xs([1, 0], level=["InitON", "Roff"], drop_level=True)
        z_df = self.get_z_magn_df(peak_z)
        magn_df["z_magn"] = z_df["magn"]
        magn_df["z_magn_stderr"] = z_df["magn_stderr"]
        return magn_df

    def get_magn_df(self, peak_xy, peak_z):
        magn_df = self.get_amp_df(peak_xy)
        z_df = self.get_z_magn_df(peak_z)
        magn_df["z_magn"] = z_df["magn"]
        magn_df["z_magn_stderr"] = z_df["magn_stderr"]
        return magn_df