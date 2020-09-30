from dataclasses import dataclass
import xarray as xr

from .image_analysis import ImageParameters
from .ion_analysis import IonParameters
from .st_state_patch import _SessionState as State
import rydanalysis as ra


class RydState(State):
    sequence_selector = 'by date'
    data = None  # xr.Dataset(coords={'shot': []})
    fit_names = ['density']
    image_parameters = ImageParameters()
    ion_parameters = IonParameters()

    def __init__(self, key=None, is_global=False):
        super(RydState, self).__init__(key, is_global)
        self.old_structure = ra.OldStructure("", initial_update=False)


