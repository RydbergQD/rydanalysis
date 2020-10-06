import xarray as xr
from .ion_seqeunce_analysis import get_ion_summary


class PostAnalysis:
    def __init__(self, data: xr.Dataset):
        self.data = data

        self.analysis_summary = get_ion_summary(data)