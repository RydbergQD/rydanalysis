import numpy as np
from lmfit import Parameters, minimize, report_fit, Model
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import pandas as pd


def params_to_dict(params):
    return pd.Series({key: params[key].value for key in params.keys()})


def dict_to_params(params_dict):
    params = Parameters()
    for key, value in params_dict.items():
        params.add(key, value)
    return params


class Fit2d:
    colors = [(1, 1, 1), to_rgb("#5597C6"), to_rgb("#60BD68"), to_rgb("#FAA43A"), to_rgb("#d37096")]

    c_map_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(c_map_name, colors, N=200)

    @staticmethod
    def _function(x, y):
        pass
    
    def __init__(self, data, x=None, y=None, params=None, extent=None):
        self.data = data
        self.extent = extent
        if self.extent is None:
            resolution = 2.08
            self.extent = np.array([-data.shape[1], data.shape[1], -data.shape[0], data.shape[0]]) * resolution / 2
        if x is None or y is None:
            self.x, self.y = self.get_mesh()

        if params is None:
            model = Model(self._function)
            params = model.make_params()

        self.params = params
        self.fit_object = None

    def fit_data(self, method='LeastSq'):
        fit_object = minimize(self.residuals, self.params, method=method)
        self.fit_object = fit_object
        self.params = fit_object.params

    def residuals(self, p):
        return self.data - self._function([self.x, self.y], **params_to_dict(p))

    def plot(self, ax, contour_color='b'):
        ax.imshow(self.data, cmap=self.cm, extent=self.extent, origin='lower')
        ax.contour(self.x, self.y, self._function([self.x, self.y], **params_to_dict(self.params)), 6,
                   colors=contour_color, linewidths=0.5)

    def get_mesh(self):
        x = np.linspace(self.extent[0], self.extent[1], np.shape(self.data)[1])
        y = np.linspace(self.extent[2], self.extent[3], np.shape(self.data)[0])
        return np.meshgrid(x, y)

    def report(self):
        print(report_fit(self.fit_object))

    def param_to_dict(self):
        return params_to_dict(self.params)


class Fit2dGaussian(Fit2d):
    def __init__(self, data, x=None, y=None, params=None, extent=None):
        super().__init__(data, x, y, params, extent)

    @staticmethod
    def _function(args, amp=1, cen_x=10, cen_y=0, sig_x=50 * 2.08, sig_y=20 * 2.08, offset=0, theta=0):
        x, y = args
        a = (np.cos(theta)**2) / (2 * sig_x**2) + (np.sin(theta)**2) / (2 * sig_y**2)
        b = -(np.sin(2 * theta)) / (4 * sig_x**2) + (np.sin(2 * theta)) / (4 * sig_y**2)
        c = (np.sin(theta)**2) / (2 * sig_x**2) + (np.cos(theta)**2) / (2 * sig_y**2)
        return amp * np.exp(-(a*(cen_x - x)**2 + 2*b*(cen_x - x)*(cen_y - y) + c*(cen_y - y)**2)) + offset


class Fit2d2Gaussian(Fit2d):
    def __init__(self, data, x=None, y=None, params=None):
        super().__init__(data, x, y, params)

    @staticmethod
    def _function(args, amp=1, cen_x=250, cen_y=50, sig_x=50, sig_y=10,
                  amp2=1, cen_x2=250, cen_y2=50, sig_x2=50, sig_y2=10,
                  offset=0):
        x, y = args
        return (amp * np.exp(-(((cen_x - x) / sig_x) ** 2 + ((cen_y - y) / sig_y) ** 2) / 2.0)
                + amp2 * np.exp(-(((cen_x2 - x) / sig_x2) ** 2 + ((cen_y2 - y) / sig_y2) ** 2) / 2.0)
                + offset)

