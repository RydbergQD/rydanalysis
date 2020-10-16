from distutils.dir_util import copy_tree

import streamlit as st
from pathlib import Path
import datetime
import pandas as pd
from dataclasses import dataclass
import xarray as xr
from typing import Tuple, List, Any
from numbers import Number

from rydanalysis.data_structure.extract_old_structure import OldStructure, load_data
from rydanalysis.data_structure.ryd_data import load_ryd_data


from streamlit_apps.st_state_patch import get as get_session_state


@dataclass()
class ImportExport:
    base_path: str = r"\\147.142.18.81\rydberg\data"
    date: Any = datetime.date.today()
    scan_name: str = ""
    export_path: str = r"\\147.142.18.81\qd-local\qd\rydberg\Projekte - Projects\2020_Aging"
    date_to_destiny: bool = True

    def run(self):
        st.markdown(
            r"""
            # Import and export data
            """
        )
        self.from_date()

    def from_date(self):
        expander = st.beta_expander("Import from date", expanded=True)
        with expander:
            import_path = self.set_import_path_from_date()
            old_structure = OldStructure(import_path)
            export_path = self.set_export_path(old_structure)
            col_save, col_load = st.beta_columns(2)
            with col_save:
                streamlit_export(old_structure, export_path)
            with col_load:
                streamlit_load_hdf5(export_path)

    def set_import_path_from_date(self):
        self.base_path = st.text_input('Enter base path', value=str(self.base_path))
        base_path = Path(self.base_path)
        if not base_path.is_dir():
            st.text("'Base path is not a valid directory. '")

        # Choose date
        self.date = st.date_input('Choose data', self.date)
        strf_date = self.date.strftime('%Y_%m_%d')
        date_path = base_path / strf_date

        # Choose scan name
        if date_path.is_dir():
            scan_names = [x.name for x in date_path.iterdir() if x.is_dir()]
        else:
            st.text(
                'No experimental run was found on that date. Consider Changing the base path or '
                'verify the date.')
            scan_names = []
        if self.scan_name in scan_names:
            scan_name_index = scan_names.index(self.scan_name)
        else:
            scan_name_index = 0

        scan_name = st.selectbox('Choose run', scan_names, index=scan_name_index)
        return date_path / scan_name

    def set_export_path(self, old_structure):
        st.write("""## Export data""")
        self.export_path = st.text_input("Save as netcdf here", value=self.export_path)
        export_path = Path(self.export_path)

        # Add date to path
        if old_structure.date is None:
            self.date_to_destiny = st.checkbox("Add date to destiny path:", self.date_to_destiny)
        else:
            self.date_to_destiny = False
        if self.date_to_destiny:
            export_path = export_path / old_structure.strf_date / old_structure.scan_name

        st.text("""Data will be saved here: {}""".format(str(export_path)))
        return export_path


def streamlit_export(import_path, export_path):
    if st.button('save to hdf5'):
        old_structure = OldStructure(import_path)
        (export_path / 'Analysis').mkdir(parents=True, exist_ok=True)
        old_structure.copy_sequences_variables(export_path)
        old_structure.save_data(export_path / "raw_data")


def streamlit_load_hdf5(path):
    load_lazy = st.checkbox("Load lazy: ", value=False)
    if st.button('load_netcdf'):
        return load_data(path / "raw_data", lazy=load_lazy)


if __name__ == '__main__':
    state = get_session_state(
        sequence_selector='by date',
        data=None,
        fit_names=['density'],
        old_structure=OldStructure(Path(r"\\147.142.18.81\rydberg\data\2020_10_07\01_PhaseScan2")),
        exporter=ImportExport()
    )
