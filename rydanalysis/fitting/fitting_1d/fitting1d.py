import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import peak_widths
from lmfit import Model
from lmfit.model import ModelResult
from lmfit.models import COMMON_INIT_DOC, COMMON_GUESS_DOC, update_param_vals
import pandas as pd
from scipy import fftpack


def fit_results_to_dict(fit_out: ModelResult):
    """Create a Series with the fit results."""
    param_list = list(fit_out.params.values())
    param_names = [param.name for param in param_list]
    param_values = [param.value for param in param_list]
    stderr_names = [param.name + "_stderr" for param in param_list]
    stderr_values = [param.stderr for param in param_list]
    fit_stat_names = ["chisqr", "redchi", "success", "errorbars"]
    fit_stat_values = [
        fit_out.chisqr,
        fit_out.redchi,
        fit_out.success,
        fit_out.errorbars,
    ]

    names = np.concatenate((param_names, stderr_names, fit_stat_names), axis=0)
    values = np.concatenate((param_values, stderr_values, fit_stat_values), axis=0)

    dictionary = {name: value for name, value in zip(names, values)}
    return pd.Series(dictionary)


def fit_params_to_lmfit_params(fit_params, model):
    params = model.make_params()
    params = fit_params[params.keys()]
    return model.make_params(**params.to_dict())


class DampedCosineModel(Model):
    """Cosine model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )

        super(DampedCosineModel, self).__init__(self._damped_cosine, **kwargs)
        self._set_paramhints_prefix()
    
    @staticmethod
    def _damped_cosine(x, freq=1.0, amp=1.0, phase=0.0, gamma=0.0, offset=0):
            return amp * np.cos(2 * np.pi * (freq * x + phase/360)) * np.exp(-gamma * x) + offset

    def _set_paramhints_prefix(self):
        self.set_param_hint("amp", min=0)
        self.set_param_hint("phase")

    def get_fft(self, data, x=None):
        if x is None:
            x = np.arange(len(data))
        # build frequency array
        n_samples = len(data)
        fs = (x[1] - x[0])
        f = rfftfreq(n_samples, fs)

        # compute fft
        fft = rfft(data) / n_samples * 2
        return f, fft

    def find_max(self, data, x=None):
        if x is None:
            x = np.arange(data)
        data2 = np.abs(data)**2
        data2[0] = 0
        fft_max_i = np.argmax(data2)

        widths, _, _, _ = peak_widths(data2, [fft_max_i])
        gamma = widths[0] / (x[1] - x[0])
        amp = np.sqrt(data2[fft_max_i])

        freq = x[fft_max_i]
        fft_max = data[fft_max_i]
        phase = np.arctan2(fft_max.imag, fft_max.real) * 360/(2*np.pi)
        return gamma, amp, freq, phase

    def guess(self, data, x=None, **kwargs):
        f, fft = self.get_fft(data, x=x)

        gamma, amp, freq, phase = self.find_max(fft, x=f)
        offset = np.mean(data)

        params = self.make_params(amp=amp, offset=offset, phase=phase, freq=freq, gamma=gamma)
        params["amp"].min = 0
        params["freq"].min = 0
        params["gamma"].min = 0
        return update_param_vals(params, self.prefix, **kwargs)

    @staticmethod
    def _get_freq(data, x=None):
        dt = 1
        if x is not None:
            dt = x[1] - x[0]
        fourier = fftpack.fft(data)
        power = np.abs(fourier)
        sample_freq = fftpack.fftfreq(data.size, dt)

        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        peak_freq = freqs[power[pos_mask].argmax()]
        return peak_freq

    @staticmethod
    def _get_phase(data, x=None):
        pass

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class CosineModel(Model):
    """Cosine model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )

        super(CosineModel, self).__init__(self._damped_cosine, **kwargs)
        self._set_paramhints_prefix()
    
    @staticmethod
    def _damped_cosine(x, freq=1.0, amp=1.0, phase=0.0, gamma=0.0, offset=0):
            return amp * np.cos(2 * np.pi * (freq * x + phase/360)) * np.exp(-gamma * x) + offset

    def _set_paramhints_prefix(self):
        self.set_param_hint("amp", min=0)
        self.set_param_hint("phase")

    def get_fft(self, data, x=None):
        if x is None:
            x = np.arange(len(data))
        # build frequency array
        n_samples = len(data)
        fs = (x[1] - x[0])
        f = rfftfreq(n_samples, fs)

        # compute fft
        fft = rfft(data) / n_samples * 2
        return f, fft

    def find_max(self, data, x=None):
        if x is None:
            x = np.arange(data)
        data2 = np.abs(data)**2
        data2[0] = 0
        fft_max_i = np.argmax(data2)

        widths, _, _, _ = peak_widths(data2, [fft_max_i])
        gamma = widths[0] / (x[1] - x[0])
        amp = np.sqrt(data2[fft_max_i])

        freq = x[fft_max_i]
        fft_max = data[fft_max_i]
        phase = np.arctan2(fft_max.imag, fft_max.real) * 360/(2*np.pi)
        return gamma, amp, freq, phase

    def guess(self, data, x=None, **kwargs):
        f, fft = self.get_fft(data, x=x)

        gamma, amp, freq, phase = self.find_max(fft, x=f)
        offset = np.mean(data)

        params = self.make_params(amp=amp, offset=offset, phase=phase, freq=freq, gamma=gamma)
        params["amp"].min = 0
        params["freq"].min = 0
        params["gamma"].min = 0
        return update_param_vals(params, self.prefix, **kwargs)

    @staticmethod
    def _get_freq(data, x=None):
        dt = 1
        if x is not None:
            dt = x[1] - x[0]
        fourier = fftpack.fft(data)
        power = np.abs(fourier)
        sample_freq = fftpack.fftfreq(data.size, dt)

        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        peak_freq = freqs[power[pos_mask].argmax()]
        return peak_freq

    @staticmethod
    def _get_phase(data, x=None):
        pass

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ExpSatModel(Model):
    """Cosine model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )

        def exponential_sat(x, amp=1.0, tau=1.0, offset=0):
            """Get model for dumped oscillations."""
            return amp * (1 - np.exp(-x / tau)) + offset

        super(ExpSatModel, self).__init__(exponential_sat, **kwargs)

    def guess(self, data, x=None, phase=0, **kwargs):
        """Estimate initial model parameter values from data."""
        amp_val = np.max(data) - np.min(data)
        offset_val = np.min(data)
        pars = self.make_params(amp=amp_val, offset=offset_val)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ExpDecayModel(Model):
    """Exponential decay model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )

        def exponential_decay(x, amp=1.0, tau=1.0, beta=1):
            """Get model for dumped oscillations."""
            return amp * np.exp(-((x / tau) ** beta))

        super(ExpDecayModel, self).__init__(exponential_decay, **kwargs)

    def guess(self, data, x=None, phase=0, **kwargs):
        """Estimate initial model parameter values from data."""
        amp_val = np.max(data) - np.min(data)
        offset_val = np.min(data)
        pars = self.make_params(amp=amp_val, offset=offset_val)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy.random as npr
    from scipy import fftpack
    from lmfit import Parameter

    def test_cosine():
        x = np.linspace(0, 1, 100)
        y = 5 * np.sin(2 * np.pi * x)  #  * np.exp(-x / 2.5)
        y += npr.choice([-1, 1], size=y.shape) * npr.random(size=y.shape) * 5

        model = CosineModel()
        params = model.guess(y, x=x)
        params["freq"] = Parameter("freq", 1, vary=False)
        fit = model.fit(y, params, x=x)

        print(fit_results_to_dict(fit))
        fit.plot()

        plt.show()

    def test_exponential_sat():
        x = np.linspace(0, 1, 100)
        amp = 2
        tau = 0.3
        offset = 0
        y = amp * (1 - np.exp(-x / tau)) + offset
        y += npr.choice([-1, 1], size=y.shape) * npr.random(size=y.shape)

        model = ExpSatModel()
        params = model.guess(y, x=x)
        params["offset"] = Parameter("offset", 0, vary=False)
        fit = model.fit(y, params, x=x)

        print(fit_results_to_dict(fit))
        fit.plot()

        plt.show()

    test_exponential_sat()
