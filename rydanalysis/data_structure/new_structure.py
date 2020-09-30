import xarray as xr
from pathlib import Path
from rydanalysis.data_structure.ryd_data import load_ryd_data


class NewStructure:
    def __init__(self, path):
        self.path = Path(path)
        self.data = load_ryd_data(path)
