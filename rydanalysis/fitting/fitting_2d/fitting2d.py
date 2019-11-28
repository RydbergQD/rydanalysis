import numpy as np
from lmfit import Parameters, minimize, report_fit, Model
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import pandas as pd
from scipy import ndimage, integrate


def params_to_dict(params):
    return pd.Series({key: params[key].value for key in params.keys()})


def dict_to_params(params_dict):
    params = Parameters()
    for key, value in params_dict.items():
        params.add(key, value)
    return params


def center_of_mass(input, labels=None, index=None):
    normalizer = np.nansum(input, labels, index)
    grids = np.ogrid[[slice(0, i) for i in input.shape]]

    results = [np.nansum(input * grids[dir].astype(float), labels, index) / normalizer
               for dir in range(input.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]


def cov_guess(image):
    nonfinite = np.where(~np.isfinite(image))
    image[nonfinite] = 0.0
    x0, y0 = ndimage.measurements.center_of_mass(image)
    if np.isnan(x0) or np.isnan(y0):
        x0 = image.shape[0] / 2
        y0 = image.shape[1] / 2
    x0 = int(x0)
    y0 = int(y0)
    offset = image.min()
    image = (image - offset)
    im_sum = image.sum()
    image = image / im_sum
    x = np.arange(0 - x0, image.shape[0] - x0)
    y = np.arange(0 - y0, image.shape[1] - y0)
    xx = x ** 2
    yy = y ** 2
    YY, XX = np.meshgrid(xx, yy, indexing='ij')
    Y, X = np.meshgrid(x, y, indexing='ij')
    try:
        cxx = (XX * image).sum()
    except:
        print(x0, x.shape, xx.shape)
    cyy = (YY * image).sum()
    cxy = (X * Y * image).sum()
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    l, v = np.linalg.eig(cov)
    theta = np.arctan2(*v[0])
    return x0, y0, np.sqrt(l[0]), np.sqrt(l[1]), offset, theta, im_sum


def restrict_to_init(pars, dev=0):
    pars = pars.copy()
    for key in pars:
        p = pars[key]
        p.min = p.value * (1.0 - dev)
        p.max = p.value * (1.0 + dev)
    return pars


class Fit2d:
    colors = [(1, 1, 1), to_rgb("#5597C6"), to_rgb("#60BD68"), to_rgb("#FAA43A"), to_rgb("#d37096")]

    c_map_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(c_map_name, colors, N=200)

    @staticmethod
    def _function(x, y):
        pass

    @classmethod
    def get_default_params(cls):
        model = Model(cls._function)
        return model.make_params()

    def __init__(self, data, params=None):
        self.data = data

        self.x, self.y = self.get_mesh()

        if params is None:
            model = Model(self._function)
            params = model.make_params()

        self.params = params
        self.fit_object = None

    def fit_data(self, method='LeastSq'):
        fit_object = minimize(self.residuals, self.params, method=method, nan_policy='omit')
        self.fit_object = fit_object
        self.params = fit_object.params
        return fit_object

    def residuals(self, p):
        return self.data - self._function([self.x, self.y], **params_to_dict(p))

    def plot(self, ax, image_kwargs=dict(), contour_kwargs=dict()):
        ax.imshow(self.data, **image_kwargs)  # origin='lower'
        # ax.pcolormesh(self.x,self.y,self.data)
        ax.contour(self._function([self.x, self.y], **params_to_dict(self.params)), 8, **contour_kwargs)

    def get_mesh(self):
        x = np.arange(self.data.shape[0])
        y = np.arange(self.data.shape[1])
        return np.meshgrid(x, y, indexing='ij')

    def report(self):
        print(report_fit(self.fit_object))

    def param_to_dict(self):
        return params_to_dict(self.params)

    def eval(self, arg, params=None):
        if params is None:
            params = self.params
        return self._function(arg, **params_to_dict(params))


class Fit2dGaussian(Fit2d):
    def __init__(self, data, params=None):
        super().__init__(data, params)

    @staticmethod
    def _function(args, amp=1, cen_x=10, cen_y=0, sig_x=50 * 2.08, sig_y=20 * 2.08, offset=0, theta=0):
        x, y = args
        a = (np.cos(theta) ** 2) / (2 * sig_x ** 2) + (np.sin(theta) ** 2) / (2 * sig_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sig_x ** 2) + (np.sin(2 * theta)) / (4 * sig_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sig_x ** 2) + (np.cos(theta) ** 2) / (2 * sig_y ** 2)
        return amp * np.exp(-(a * (cen_x - x) ** 2 + 2 * b * (cen_x - x) * (cen_y - y) + c * (cen_y - y) ** 2)) + offset

    def guess(self, data):
        cen_x, cen_y, sig_x, sig_y, offset, theta, im_sum = cov_guess(data)

        def func(x, y, *params, **kwargs):
            return self._function([x, y], *params, **kwargs)

        dibu_int, err = integrate.nquad(func, [[0, data.shape[0]], [0, data.shape[1]]],
                                        args=(1, cen_x, cen_y, sig_x, sig_y, offset, theta))
        amp = im_sum / dibu_int

        pars = Parameters()  # Model(self._function).make_params(amp=amp, cen_x=cen_x, cen_y=cen_y, sig_x=sig_x, sig_y=sig_y, offset=offset, theta=theta)
        pars.add('amp', value=amp)
        pars.add('cen_x', value=cen_x)
        pars.add('cen_y', value=cen_y)
        pars.add('sig_x', value=sig_x)
        pars.add('sig_y', value=sig_y)
        pars.add('theta', value=theta)
        pars.add('offset', value=offset)
        return pars
        # return update_param_vals(pars, self.prefix, **kwargs)


class Fit2d2Gaussian(Fit2d):
    def __init__(self, data, params=None):
        super().__init__(data, params)

    @staticmethod
    def _function(args, amp1, cen_x1, cen_y1, sig_x1, sig_y1, theta1, amp2, cen_x2, cen_y2, sig_x2, sig_y2, theta2,
                  offset):
        x, y = args
        # cen_x1 = float(cen_x1)
        # cen_y1 = float(cen_y1)
        # cen_x2 = float(cen_x2)
        # cen_y2 = float(cen_y2)

        a1 = (np.cos(theta1) ** 2) / (2 * sig_x1 ** 2) + (np.sin(theta1) ** 2) / (2 * sig_y1 ** 2)
        b1 = -(np.sin(2 * theta1)) / (4 * sig_x1 ** 2) + (np.sin(2 * theta1)) / (4 * sig_y1 ** 2)
        c1 = (np.sin(theta1) ** 2) / (2 * sig_x1 ** 2) + (np.cos(theta1) ** 2) / (2 * sig_y1 ** 2)
        a2 = (np.cos(theta2) ** 2) / (2 * sig_x2 ** 2) + (np.sin(theta2) ** 2) / (2 * sig_y2 ** 2)
        b2 = -(np.sin(2 * theta2)) / (4 * sig_x2 ** 2) + (np.sin(2 * theta2)) / (4 * sig_y2 ** 2)
        c2 = (np.sin(theta2) ** 2) / (2 * sig_x2 ** 2) + (np.cos(theta2) ** 2) / (2 * sig_y2 ** 2)

        g = offset + amp1 * np.exp(- (a1 * ((x - cen_x1) ** 2) + 2 * b1 * (x - cen_x1) * (y - cen_y1) + c1 * (
                    (y - cen_y1) ** 2))) + amp2 * np.exp(
            - (a2 * ((x - cen_x2) ** 2) + 2 * b2 * (x - cen_x2) * (y - cen_y2) + c2 * ((y - cen_y2) ** 2)))

        return g

# class Fit2d2Gaussian(Fit2d):
#     def __init__(self, data, x=None, y=None, params=None):
#         super().__init__(data, x, y, params)

#     @staticmethod
#     def _function(args, amp=1, cen_x=250, cen_y=50, sig_x=50, sig_y=10,
#                   amp2=1, cen_x2=250, cen_y2=50, sig_x2=50, sig_y2=10,
#                   offset=0):
#         x, y = args
#         return (amp * np.exp(-(((cen_x - x) / sig_x) ** 2 + ((cen_y - y) / sig_y) ** 2) / 2.0)
#                 + amp2 * np.exp(-(((cen_x2 - x) / sig_x2) ** 2 + ((cen_y2 - y) / sig_y2) ** 2) / 2.0)
#                 + offset)
