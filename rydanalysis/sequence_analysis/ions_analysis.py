from scipy.signal import find_peaks, convolve
from scipy.signal.wavelets import cwt, ricker
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm


def _count_ions_reduce(trace, axis=0, height=0.03, width=3, **kwargs):
    """
    Count ions on a single trace using scipy.find_peaks.
    Args:
        trace: 1D array
        axis: not used but needed to use this function with xarray.reduce
        height: minimal height of teh peaks
        width: minimal width of the peaks (in pixels)
        **kwargs: additional kwargs given to scipy.find_peaks

    Returns:
        number of the peaks
    """
    peaks, _ = find_peaks(-trace, height=height, width=width, **kwargs)
    return len(peaks)


def count_ions(scope_traces, dim='time', height=0.03, width=3, **kwargs):
    """
    Count ions on multiple traces bundled in an xarray DataArray or Dataset using scipy.find_peaks
    Args:
        scope_traces: xarray DataArray or Dataset
        dim: Dimension along the peaks are counted
        height: minimal height of teh peaks
        width: minimal width of the peaks (in pixels)
        **kwargs: additional kwargs given to scipy.find_peaks

    Returns:
        Reduced DataArray or Dataset with
    """
    kwargs.update(height=height, width=width)
    group_by_object = scope_traces.groupby('shot')
    ions = group_by_object.reduce(_count_ions_reduce, dim=dim, **kwargs)
    if isinstance(scope_traces, xr.DataArray):
        ions.name = 'ions'
    return ions


def _integrate_ions_reduce(trace, axis=0, height=0.03):
    """
    Integrates ion trace (only accounting values larger than height.
    Args:
        trace: 1D array
        axis: not used but needed to use this function with xarray.reduce
        height: minimal height of teh peaks

    Returns:
        Integrated ion signal (float)
    """
    trace = -trace[trace < -height]
    return trace.sum()


def integrate_ions(scope_traces, dim='time', height=0.03):
    """
    Integrate ions on multiple traces bundled in an xarray DataArray or Dataset.
    Args:
        scope_traces: xarray DataArray or Dataset
        dim: Dimension along the peaks are counted
        height: minimal height of teh peaks

    Returns:
        Integrated ion signal (xarray DataArray or Dataset)
    """
    group_by_object = scope_traces.groupby('tmstp')
    ions = group_by_object.reduce(_count_ions_reduce, dim=dim, height=height)
    if isinstance(scope_traces, xr.DataArray):
        ions.name = 'ionsInt'
    return ions


def cwt_xr(trace, wavelet, widths):
    time_scale = extract_time_scale(trace.time)

    widths_in_pixels = widths / time_scale
    wavelet_transform = cwt(trace, wavelet, widths_in_pixels)
    return xr.DataArray(
        wavelet_transform,
        coords={'time': trace.time, 'width': widths},
        dims=['width', 'time']
    )


def pixel_to_time(pixel, time_scale):
    if pixel:
        return pixel / time_scale


def extract_time_scale(time):
    time_values = np.sort(time.values)
    return time_values[1] - time_values[0]


def find_peaks_xr(trace, height=None, prominence=None, threshold=None, distance=None, width=None):
    time_scale = extract_time_scale(trace.time)

    peaks_index, properties = find_peaks(
        trace,
        height=height,
        distance=pixel_to_time(distance, time_scale),
        width=pixel_to_time(width, time_scale),
        prominence=prominence,
        threshold=threshold
    )
    return trace[peaks_index]


def convolve_wavelet(trace, wavelet, width):
    length = min(10 * width, len(trace))
    return convolve(trace, wavelet(length, width), mode='same')


def convolve_wavelet_xr(trace, wavelet, width):
    time_scale = extract_time_scale(trace.time)
    width_in_pixels = width / time_scale

    wavelet = convolve_wavelet(trace, wavelet, width_in_pixels)
    return xr.DataArray(
        wavelet,
        coords=trace.coords,
        dims=trace.dims
    )


def find_peaks_wavelet(trace, width=0.8, prominence=0.014):
    transformed = convolve_wavelet_xr(trace, wavelet=ricker, width=width)
    return find_peaks_xr(transformed, prominence=prominence)


def find_all_peaks_wavelet(traces, width=0.8, prominence=0.014):
    all_peaks = xr.zeros_like(traces, dtype=np.dtype(bool))
    for coord, trace in tqdm(traces.groupby('run')):
        trace = trace.squeeze()
        trace = trace.squeeze()
        peaks = find_peaks_wavelet(trace, width=width, prominence=prominence)
        all_peaks.loc[peaks.coords] = True
    return all_peaks


def find_all_peaks_scipy(traces, width=0.8, height=0.0005, prominence=0.001, distance=2):
    all_peaks = xr.zeros_like(traces, dtype=np.dtype(bool))
    for coord, trace in tqdm(traces.groupby('run')):
        trace = trace.squeeze()
        trace = trace.squeeze()
        peaks = find_peaks_xr(trace, width=width, height=height, distance=distance, prominence=prominence)
        all_peaks.loc[peaks.coords] = True
    return all_peaks
