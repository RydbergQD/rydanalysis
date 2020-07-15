from scipy.signal import find_peaks, convolve
from scipy.signal.wavelets import cwt, ricker
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm


@xr.register_dataarray_accessor("peaks_summary")
class PeaksSummaryAccessor:
    def __init__(self, xarray_obj):
        self.traces = xarray_obj

    def count(self, dim='time', height=0.03, width=3, **kwargs):
        """
        Count peaks using scipy.find_peaks
        Args:
            dim: Dimension along the peaks are counted
            height: minimal height of teh peaks
            width: minimal width of the peaks (in pixels)
            **kwargs: additional kwargs given to scipy.find_peaks

        Returns:
            Reduced DataArray or Dataset with the number of counts
        """
        kwargs.update(height=height, width=width)
        group_by_object = self.traces.groupby('shot')

        def _count_reduce(trace, axis=0, height=0.03, width=3, **kwargs):
            peaks, _ = find_peaks(-trace, height=height, width=width, **kwargs)
            return len(peaks)
        counts = group_by_object.reduce(_count_reduce, dim=dim, **kwargs)
        counts.name = 'ion_counts'
        return counts

    def integrate(self, dim='time', height=0.03):
        """
        Integrate ions on multiple traces bundled in an xarray DataArray or Dataset.
        Args:
            dim: Dimension along the peaks are integrated
            height: minimal height of teh peaks

        Returns:
            Integrated ion signal (xarray DataArray or Dataset)
        """
        traces = -self.traces[self.traces < -height]
        integral = traces.groupby('shot').sum()
        integral.name = 'ionsInt'
        return integral

    def find_all_peaks_wavelet(self, width=0.8, prominence=0.014):
        """
        Find all peak positions using the wavelet peak finding algorithm.
        Args:
            width: minimal width of the peaks (in s)
            prominence: minimal prominence

        Returns:
            Data Array with True a the location of the peaks, False if no peak was found.
        """
        traces = self.traces
        all_peaks = xr.zeros_like(traces, dtype=np.dtype(bool))
        for coord, trace in tqdm(traces.groupby('shot')):
            trace = trace.squeeze()
            peaks = trace.peaks.find_peaks_wavelet(width=width, prominence=prominence)
            all_peaks.loc[peaks.coords] = True
        return all_peaks

    def find_all_peaks_scipy(self, width=0.8, height=0.0005, prominence=0.001, distance=2):
        """
        Find all peak positions using the scipy peak finding algorithm.
        Args:
            width: minimal width of the peaks (in s)
            height: minimal height of the peaks
            prominence: minimal prominence
            distance: minimal distance between the peaks (in s)

        Returns:
            Data Array with True a the location of the peaks, False if no peak was found.
        """
        traces = self.traces
        all_peaks = xr.zeros_like(traces, dtype=np.dtype(bool))
        for coord, trace in tqdm(traces.groupby('run')):
            trace = trace.squeeze()
            peaks = trace.peaks.find_peaks(trace, width=width, height=height, distance=distance, prominence=prominence)
            all_peaks.loc[peaks.coords] = True
        return all_peaks


@xr.register_dataarray_accessor("peaks")
class PeaksAccessor:
    def __init__(self, xarray_obj):
        self.trace = xarray_obj

    def cwt_xr(self, wavelet, widths):
        widths_in_pixels = widths / self.time_scale
        wavelet_transform = cwt(self.trace, wavelet, widths_in_pixels)
        return xr.DataArray(
            wavelet_transform,
            coords={'time': self.trace.time, 'width': widths},
            dims=['width', 'time']
        )

    @property
    def time_scale(self):
        time_values = np.sort(self.trace.time.values)
        return time_values[1] - time_values[0]

    def find_peaks(self, height=None, prominence=None, threshold=None, distance=None, width=None):
        peaks_index, properties = find_peaks(
            self.trace,
            height=height,
            distance=self.time_to_pixel(distance),
            width=self.time_to_pixel(width),
            prominence=prominence,
            threshold=threshold
        )
        return self.trace[peaks_index]

    def convolve_wavelet(self, wavelet, width):
        width_in_pixels = width / self.time_scale

        wavelet = convolve_wavelet(self.trace, wavelet, width_in_pixels)
        return xr.DataArray(
            wavelet,
            coords=self.trace.coords,
            dims=self.trace.dims
        )

    def find_peaks_wavelet(self, width=0.8, prominence=0.014):
        transformed = self.convolve_wavelet(wavelet=ricker, width=width)
        return transformed.peaks.find_peaks(prominence=prominence)

    def time_to_pixel(self, time):
        """Returns a given time in pixels. if time is None, return None"""
        if time:
            return time / self.time_scale


def convolve_wavelet(trace, wavelet, width):
    length = min(10 * width, len(trace))
    return convolve(trace, wavelet(length, width), mode='same')
