# imports
import numpy as np
import matplotlib.pyplot as plt
import rydanalysis as ra


# Define function
def gaussian(x, y, amp=1, cen_x=10, cen_y=0, sig_x=50 * 2.08, sig_y=20 * 2.08, offset=0, theta=0):
    a = (np.cos(theta) ** 2) / (2 * sig_x ** 2) + (np.sin(theta) ** 2) / (2 * sig_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sig_x ** 2) + (np.sin(2 * theta)) / (4 * sig_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sig_x ** 2) + (np.cos(theta) ** 2) / (2 * sig_y ** 2)
    return amp * np.exp(-(a * (cen_x - x) ** 2 + 2 * b * (cen_x - x) * (cen_y - y) + c * (cen_y - y) ** 2)) + offset


# Create test data
x = np.linspace(-100, 100, 201)
y = np.linspace(-50, 50, 401)
xx, yy = np.meshgrid(x, y, indexing='ij')
image = gaussian(xx, yy, amp=10, cen_x=-20, cen_y=0, sig_x=50, sig_y=60) + np.random.rand(len(x), len(y))
weights = np.random.rand(len(x), len(y))

# Fit the data
model = ra.Model2d(gaussian)
params = model.make_params(cen_x=20)
fit_result = model.fit(image, params, x=x, y=y, weights=weights)

# Plotting
#fit_result.plot()
#plt.show()

# Alternative using xarray
import xarray as xr

image = xr.Dataset(
    {
        'signal': (['x', 'y'], image),
        'noise': (['x', 'y'], weights)
    },
    coords={
        'x': x,
        'y':y
    }
)

fit_result = model.fit(image.signal, weights=image.noise)

# Plotting
fit_result.plot()
plt.show()