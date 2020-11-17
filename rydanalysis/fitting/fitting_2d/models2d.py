from rydanalysis.fitting.fitting_2d.model2d import Model2d

from lmfit.models import COMMON_INIT_DOC, COMMON_GUESS_DOC, update_param_vals
import numpy as np
from skimage.measure import moments, moments_central


class Gaussian2D(Model2d):
    """Constant model, with a single Parameter: ``c``.
    Note that this is 'constant' in the sense of having no dependence on
    the independent variable ``x``, not in the sense of being non-varying.
    To be clear, ``c`` will be a Parameter that will be varied
    in the fit (by default, of course).
    """

    def __init__(
        self, independent_vars=("x", "y"), prefix="", nan_policy="raise", **kwargs
    ):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )

        def gaussian(
            x,
            y,
            amp=1,
            cen_x=10,
            cen_y=0,
            sig_x=50 * 2.08,
            sig_y=20 * 2.08,
            offset=0,
            theta=0,
        ):
            # x, y = args
            a = (np.cos(theta) ** 2) / (2 * sig_x ** 2) + (np.sin(theta) ** 2) / (
                2 * sig_y ** 2
            )
            b = -(np.sin(2 * theta)) / (4 * sig_x ** 2) + (np.sin(2 * theta)) / (
                4 * sig_y ** 2
            )
            c = (np.sin(theta) ** 2) / (2 * sig_x ** 2) + (np.cos(theta) ** 2) / (
                2 * sig_y ** 2
            )
            return (
                amp
                * np.exp(
                    -(
                        a * (cen_x - x) ** 2
                        + 2 * b * (cen_x - x) * (cen_y - y)
                        + c * (cen_y - y) ** 2
                    )
                )
                + offset
            )

        Model2d.__init__(self, gaussian, **kwargs)

    def guess(self, data, use_quantile=False, **kwargs):
        """Estimate initial model parameter values from data."""
        kwargs = self._update_kwargs(data, **kwargs)
        data = self.create_xarray(data, **kwargs)
        rescaler_x = self.get_rescaler(data, "x")
        rescaler_y = self.get_rescaler(data, "y")

        cen_x, cen_y, sig_x, sig_y, theta = self.get_image_properties_from_moments_cart(
            data.values
        )

        if np.isnan(sig_y):
            sig_y = sig_x

        pars = self.make_params()
        pars["%scen_x" % self.prefix].set(value=rescaler_x(cen_x))
        pars["%scen_y" % self.prefix].set(value=rescaler_y(cen_y))
        pars["%ssig_x" % self.prefix].set(value=abs(rescaler_x(sig_x)), min=0)
        pars["%ssig_y" % self.prefix].set(value=abs(rescaler_x(sig_y)), min=0)
        theta = np.sin(
            (rescaler_y(1) - rescaler_y(0))
            / (rescaler_x(1) - rescaler_x(0))
            * np.arcsin(theta)
        )
        if use_quantile:
            amp = data.quantile(0.9, dim=("x", "y")).data
            offset = data.quantile(0.1, dim=("x", "y")).data
        else:
            amp = self.average_center(data, prefix=self.prefix, **pars)
            offset = self.average_outside(data, prefix=self.prefix, **pars)
        pars["%stheta" % self.prefix].set(
            value=theta, min=theta - np.pi / 4, max=theta + np.pi / 4
        )
        pars["%samp" % self.prefix].set(value=amp - offset)
        pars["%soffset" % self.prefix].set(value=offset)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC

    @staticmethod
    def get_image_properties_from_moments_cart(data):
        data = np.asarray(data)
        m = moments(data, 2)
        cen_x, cen_y = m[1, 0] / m[0, 0], m[0, 1] / m[0, 0]
        m_central = moments_central(data, center=(cen_x, cen_y), order=3)
        mu = m_central / m_central[0, 0]
        cov = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]])

        sig_x, sig_y = np.sqrt(np.abs(np.linalg.eigvals(cov)))
        theta = (
            1 / 2 * np.arctan(2 * m_central[1, 1] / (m_central[2, 0] - m_central[0, 2]))
        )
        return cen_x, cen_y, sig_x, sig_y, theta

    @staticmethod
    def get_rescaler(data, coord):
        max_x = float(data[coord].max())
        min_x = float(data[coord].min())
        size = data[coord].size
        return lambda n: ((size - n) * min_x + n * max_x) / size

    @staticmethod
    def average_center(data, prefix=None, **kwargs):
        # remove prefix
        try:
            kwargs = {key[len(prefix) :]: kwargs[key] for key in kwargs}
        except TypeError:
            pass
        cen_x = kwargs["cen_x"]
        cen_y = kwargs["cen_y"]
        sig_x = kwargs["sig_x"]
        sig_y = kwargs["sig_y"]
        mask = ((data.x - cen_x) / sig_x) ** 2 + (
            (data.y - cen_y) / (sig_y / 2)
        ) ** 2 <= 0.5 ** 2
        if not bool(mask.any()):
            return float(data.sel(x=cen_y.value, y=cen_y.value, method="nearest"))
        return float(data.where(mask).mean())

    @staticmethod
    def average_outside(data, prefix=None, **kwargs):
        # remove prefix
        try:
            kwargs = {key[len(prefix) :]: kwargs[key] for key in kwargs}
        except TypeError:
            pass
        cen_x = kwargs["cen_x"]
        cen_y = kwargs["cen_y"]
        sig_x = kwargs["sig_x"]
        sig_y = kwargs["sig_y"]
        mask = ((data.x - cen_x) / sig_x) ** 2 + (
            (data.y - cen_y) / (sig_y / 2)
        ) ** 2 >= 2 ** 2
        if not bool(mask.any()):
            return 0
        return float(data.where(mask).mean())

    def copy(self, **kwargs):
        """DOES NOT WORK."""
        raise NotImplementedError("Model.copy does not work. Make a new Model")
