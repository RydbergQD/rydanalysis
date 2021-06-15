from scipy import optimize
from functools import cached_property
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
class Image:
    def __init__(self, data, x=None, y=None):
        self.data = data.transpose("y", "x")

    @property
    def scale(self):
        return self.data.x[1] - self.data.x[0], self.data.y[1] - self.data.y[0]

    @property
    def mask(self):
        return ~self.data.isnull()

    @property
    def meshgrid(self):
        X, Y = np.meshgrid(self.data.x, self.data.y)
        mask = self.mask
        return X[mask], Y[mask]

    @cached_property
    def moments(self):
        data = self.data
        total = float(self.data.sum())
        center_x = float((data * data.x).sum() / data.sum())
        center_y = float((data * data.y).sum() / data.sum())
        row = self.data.sel(x=center_x, method="nearest")
        col = self.data.sel(y=center_y, method="nearest")
        sig_x = float(np.sqrt(abs((data.x - center_x)**2 * col).sum()/col.sum())) / 2
        sig_y = float(np.sqrt(abs((data.y - center_y)**2 * row).sum()/row.sum())) / 2
        height = np.nanmax(self.data)
        return pd.Series(
            {
                "height": height,
                "center_x": center_x,
                "center_y": center_y,
                "sig_x": sig_x,
                "sig_y": sig_y,
                "total": total
            }
            )

    @cached_property
    def fitgaussian(self):
        params = self.moments
        # for name, param in kwargs.items():
        #     params[name] = param
        data = self.data.values
        x, y = self.meshgrid
        data = data[self.mask.values]
        def _errorfunction(p):
            return gaussian(y, x, *p) - data
        params = params[:-1]
        p, success = optimize.leastsq(_errorfunction, params)
        result =  pd.Series(p, index=params.index)
        result["N"] = get_atom_number(result["height"], result["sig_x"], result["sig_y"])
        result["density3d"] = get_3d_density(result["height"], result["sig_x"], result["sig_y"])
        return result

    def plot_fit(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        self.data.plot(ax=ax)
        fit = gaussian(self.data.y, self.data.x, **self.fitgaussian[:-2])
        fit.plot.contour(ax=ax)
        
        ax.set_aspect("equal")
        plt.text(0.95, 0.05, f"""
            x : {self.fitgaussian["center_x"]}
            y : {self.fitgaussian["center_y"]}
            sig_x : {self.fitgaussian["sig_x"]}
            sig_y : {self.fitgaussian["sig_y"]}""",
                    fontsize=5, horizontalalignment='right',
                    verticalalignment='bottom', transform=ax.transAxes)
        return ax

    def plot_residuals(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        fit = gaussian(self.data.y, self.data.x, **self.fitgaussian[:-2])
        residuals = self.data - fit
        residuals.plot(ax=ax)
        
        ax.set_aspect("equal")
        plt.text(0.95, 0.05, f"""
            x : {self.fitgaussian["center_x"]}
            y : {self.fitgaussian["center_y"]}
            sig_x : {self.fitgaussian["sig_x"]}
            sig_y : {self.fitgaussian["sig_y"]}""",
                    fontsize=5, horizontalalignment='right',
                    verticalalignment='bottom', transform=ax.transAxes)
        return ax

def get_atom_number(height, sig_x, sig_y, unit='um'):
    return 2 * np.pi *  np.abs(height * sig_x * sig_y) * 1e-12

def get_3d_density(amp, sig_x, sig_y, unit='um'):
    N = get_atom_number(amp, sig_x, sig_y, unit)
    sig_short, sig_long = np.sort([sig_x, sig_y])
    sig_long = np.sqrt(2*sig_long**2 - sig_short**2)
    return 1/(np.sqrt((2*np.pi)**3)*sig_short**2*sig_long) * N * 1e12


def gaussian(x, y, height=0, center_x=0, center_y=0, sig_x=0, sig_y=0):
    """Returns a gaussian function with the given parameters"""
    return height*np.exp(
                -(center_x-x)**2/(2 * sig_x**2) - (center_y-y)**2/(2*sig_y**2)
                )

