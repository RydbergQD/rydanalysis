import numpy as np
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
    stderr_names = [param.name + '_stderr' for param in param_list]
    stderr_values = [param.stderr for param in param_list]
    fit_stat_names = ['chisqr', 'redchi', 'success', 'errorbars']
    fit_stat_values = [fit_out.chisqr, fit_out.redchi, fit_out.success, fit_out.errorbars]

    names = np.concatenate((param_names, stderr_names, fit_stat_names), axis=0)
    values = np.concatenate(
        (param_values, stderr_values, fit_stat_values), axis=0)

    dictionary = {name: value for name, value in zip(names, values)}
    return pd.Series(dictionary)


def fit_params_to_lmfit_params(fit_params, model):
    params = model.make_params()
    params = fit_params[params.keys()]
    return model.make_params(**params.to_dict())


class CosineModel(Model):
    """Cosine model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def cosine(x, freq=1.0, amp=1.0, phase=0.0, offset=0.0):
            return amp * np.cos(2 * np.pi * freq * (x + phase)) + offset
        super(CosineModel, self).__init__(cosine, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('amp', min=0)
        self.set_param_hint('phase')

    def guess(self, data, x=None, phase=0, **kwargs):
        """Estimate initial model parameter values from data."""
        amp_val = (np.max(data) - np.min(data))/2
        offset_val = np.mean(data)
        pars = self.make_params(amp=amp_val, offset=offset_val)
        if x is not None:
            freq_val = self._get_freq(data, x)
            pars = self.make_params(amp=amp_val, offset=offset_val, freq=freq_val)

            pars['amp'].vary = False
            pars['offset'].vary = False
            pars['freq'].vary = False
            initial_fit = self.fit(data, pars, x=x, method='powell')
            pars = initial_fit.params
            pars['amp'].vary = True
            pars['offset'].vary = True
            pars['freq'].vary = True
        return update_param_vals(pars, self.prefix, **kwargs)

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

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class DampedCosineModel(Model):
    """Cosine model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def damped_cosine(x, freq=1.0, amp=1.0, phase=0.0, offset=0.0, decay=0):
            return 0.5 * amp * (np.cos(2 * np.pi * freq * x + 2 * np.pi * phase) * np.exp(-x * decay)) + offset

        super(DampedCosineModel, self).__init__(damped_cosine, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('amp', min=0)
        self.set_param_hint('phase')

    def guess(self, data, x=None, phase=0, **kwargs):
        """Estimate initial model parameter values from data."""
        amp_val = (np.max(data) - np.min(data))
        offset_val = np.mean(data)
        pars = self.make_params(amp=amp_val, offset=offset_val)
        if x is not None:
            freq_val = self._get_freq(data, x)
            pars = self.make_params(amp=amp_val, offset=offset_val, freq=freq_val, decay=0)
            pars['amp'].vary = False
            pars['offset'].vary = False
            pars['freq'].vary = False
            pars['decay'].vary = False
            initial_fit = self.fit(data, pars, x=x, method='powell')
            pars = initial_fit.params
            pars['amp'].vary = True
            pars['offset'].vary = True
            pars['freq'].vary = True

            pars['decay'].vary = True
        return update_param_vals(pars, self.prefix, **kwargs)

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

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ExpSatModel(Model):
    """Cosine model, with Parameters: ``freq``, ``amp``, ``phase``, ``offset``."""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def exponential_sat(x, amp=1., tau=1., offset=0):
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

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def exponential_decay(x, amp=1., tau=1., beta=1):
            """Get model for dumped oscillations."""
            return amp * np.exp(-(x / tau)**beta)

        super(ExpDecayModel, self).__init__(exponential_decay, **kwargs)

    def guess(self, data, x=None, phase=0, **kwargs):
        """Estimate initial model parameter values from data."""
        amp_val = np.max(data) - np.min(data)
        offset_val = np.min(data)
        pars = self.make_params(amp=amp_val, offset=offset_val)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy.random as npr
    from scipy import fftpack
    from lmfit import Parameter

    def test_cosine():
        x = np.linspace(0, 1, 100)
        y = 5 * np.sin(2*np.pi * x)#  * np.exp(-x / 2.5)
        y += npr.choice([-1, 1], size=y.shape) * npr.random(size=y.shape) * 5

        model = CosineModel()
        params = model.guess(y, x=x)
        params['freq'] = Parameter('freq', 1, vary=False)
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
        params['offset'] = Parameter('offset', 0, vary=False)
        fit = model.fit(y, params, x=x)

        print(fit_results_to_dict(fit))
        fit.plot()

        plt.show()

    test_exponential_sat()