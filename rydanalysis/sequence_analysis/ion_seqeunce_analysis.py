import xarray as xr


def get_ion_summary(data):
    ion_summary = xr.Dataset()
    ion_summary["ion_counts"] = data.scope_traces.peaks_summary.count()
    ion_summary["ion_int"] = data.scope_traces.peaks_summary.integrate()
    return ion_summary