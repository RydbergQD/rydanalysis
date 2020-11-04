from scipy.signal import find_peaks, convolve
from scipy.signal.wavelets import cwt, ricker
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


@xr.register_dataarray_accessor("peaks_summary")
class PeaksSummaryAccessor:
    def __init__(self, xarray_obj):
        self.traces = xarray_obj

    def count(self, dim='time', height=0.03, width=3, sign=-1, **kwargs):
        """
        Count peaks using scipy.find_peaks
        Args:
            dim: Dimension along the peaks are counted
            height: minimal height of teh peaks
            width: minimal width of the peaks (in pixels)
            sign: -1 to detect minima
            **kwargs: additional kwargs given to scipy.find_peaks

        Returns:
            Reduced DataArray or Dataset with the number of counts
        """
        traces: xr.DataArray = self.traces
        kwargs.update(height=height, width=traces.peaks.time_to_pixel(width))
        variable_dim = [a for a in self.traces.dims if a != "time"][0]
        group_by_object = self.traces.groupby(variable_dim)

        def _count_reduce(trace, axis=0):
            peaks = trace.peaks.find_peaks(**kwargs)
            return len(peaks)
        counts = group_by_object.reduce(_count_reduce, dim=dim, **kwargs, sign=sign)
        counts.name = 'ion_counts'
        return counts

    def integrate(self, dim='time', height=0.03, sign=-1):
        """
        Integrate ions on multiple traces bundled in an xarray DataArray or Dataset.
        Args:
            dim: Dimension along the peaks are integrated
            height: minimal height of teh peaks
            sign: multiply with traces

        Returns:
            Integrated ion signal (xarray DataArray or Dataset)
        """
        traces = sign * self.traces
        traces = traces.where(traces > height)
        variable_dim = traces.ryd_data.shot_or_tmstp
        integral = traces.groupby(variable_dim).sum(dim)
        integral.name = 'ionsInt'
        return integral

    def get_peak_description(self, height=0.007, prominence=None, threshold=None, distance=None, width=2e-9, sign=-1):
        traces = self.traces
        index = traces.ryd_data.index
        peak_df = pd.DataFrame()
        for shot in tqdm(index):
            trace = traces.sel({traces.ryd_data.shot_or_tmstp: shot})
            if len(index.names) == 1:
                shot = [shot]
            df = trace.peaks.get_peak_description(height, prominence, threshold, distance, width, sign=-1)
            df.index = pd.MultiIndex.from_tuples([[i, *shot] for i in range(df.shape[0])],
                                                 names=["peak_number", *index.names])
            peak_df = peak_df.append(df)
        return peak_df


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

    def integrate(self, height=0.03, sign=-1):
        """
        Integrate ions on multiple traces bundled in an xarray DataArray or Dataset.
        Args:
            height: minimal height of teh peaks
            sign: -1 to detect minima, else +1

        Returns:
            Integrated ion signal (xarray DataArray or Dataset)
        """
        trace = sign * self.trace
        return trace.where(trace > height).sum()

    @property
    def time_scale(self):
        time_values = np.sort(self.trace.time.values)
        return time_values[1] - time_values[0]

    def _find_peaks(self, height=0, prominence=0, threshold=0, distance=0, width=0, sign=-1):
        trace = sign * self.trace
        return find_peaks(
            trace,
            height=height,
            distance=self.time_to_pixel(distance),
            width=self.time_to_pixel(width),
            prominence=prominence,
            threshold=threshold
        )

    def find_peaks(self, height=0, prominence=0, threshold=0, distance=0, width=0):
        peaks_index, properties = self._find_peaks(height, prominence, threshold, distance, width)
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
        if time is not None:
            return time / self.time_scale

    def pixel_to_time(self, index):
        """Returns a given time in pixels. if time is None, return None"""
        return index * self.time_scale

    def get_peak_description(self, height=0, prominence=0, threshold=0, distance=0, width=0, sign=-1):
        peaks_index, properties = self._find_peaks(height, prominence, threshold, distance, width, sign=sign)
        description = pd.DataFrame(properties)
        for prop in ["left_bases", "right_bases", "widths", "left_ips", "right_ips"]:
            if prop in description:
                description[prop] *= self.time_scale
        description["peak_time"] = self.pixel_to_time(peaks_index)
        return description


def convolve_wavelet(trace, wavelet, width):
    length = min(10 * width, len(trace))
    return convolve(trace, wavelet(length, width), mode='same')


def summarize_peak_description(peak_df: pd.DataFrame):
    groupby = peak_df.groupby([var for var in peak_df.index.names if var != "peak_number"])
    summary = groupby.mean()
    summary.index.name = "shot"
    summary = xr.DataArray(summary.values, dims=["shot", "variable"],
                           coords=dict(shot=summary.index, variable=summary.columns))
    summary = summary.to_dataset("variable")

    counts = groupby.apply(len)
    counts.index.name = "shot"
    summary["counts"] = counts
    return summary
