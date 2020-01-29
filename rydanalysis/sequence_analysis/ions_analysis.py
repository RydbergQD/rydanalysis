from scipy.signal import find_peaks
import xarray as xr


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
    group_by_object = scope_traces.groupby('tmstp')
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
