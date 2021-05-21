from scipy import optimize
from functools import cached_property
from mpl_toolkits.axes_grid1 import make_axes_locatable
 
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
        width_x = float(np.sqrt(((data.x - center_x)**2 * col).sum()/col.sum()))
        # width_y = np.sqrt(np.nansum(np.abs((np.arange(row.size)-center_y)**2*row))/np.nansum(row))
        width_y = float(np.sqrt(((data.y - center_y)**2 * row).sum()/row.sum()))
        height = np.nanmax(self.data)
        return pd.Series(
            {
                "height": height,
                "center_x": center_x,
                "center_y": center_y,
                "width_x": width_x,
                "width_y": width_y,
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
        return pd.Series(p, index=params.index)

    def plot_fit(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        self.data.plot(ax=ax)
        fit = gaussian(self.data.y, self.data.x, **self.fitgaussian)
        fit.plot.contour(ax=ax)
        
        ax.set_aspect("equal")
        return fig, ax


def gaussian(x, y, height=0, center_x=0, center_y=0, width_x=0, width_y=0):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)