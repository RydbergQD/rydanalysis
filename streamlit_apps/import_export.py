import streamlit as st
from pathlib import Path
import datetime
from dataclasses import dataclass, field
from typing import Any

from rydanalysis.data_structure.extract_old_structure import OldStructure, load_data


def default_path():
    base_path = Path(r"\\147.142.18.81\rydberg\data")
    date = datetime.date.today()
    strf_date = date.strftime('%Y_%m_%d')
    scan_name = ""
    return str(base_path / strf_date / scan_name)


class OldStructureStreamlit(OldStructure):
    interface: str = "streamlit"
    use_date: bool = True
    use_scan_name: bool = True
    export_path: str = r"\\147.142.18.81\qd-local\qd\rydberg\Projekte - Projects\2020_Aging"
    date_to_destiny: bool = True

    def import_export(self):
        st.markdown("### Set options")
        expander = st.beta_expander("Set import options", expanded=True)
        with expander:
            self.streamlit_set_path()

        expander = st.beta_expander("Set export options", expanded=True)
        with expander:
            self.streamlit_set_export_path()
        st.markdown("### Save and load data")
        col_save, col_load = st.beta_columns(2)
        with col_save:
            st.markdown("# ")
            self.streamlit_save(self.export_path)
        with col_load:
            return self.streamlit_load_hdf5()

    def streamlit_set_path(self):
        path = st.text_input('Enter base path', value=str(self.base_path))
        path = Path(path)
        if not path.is_dir():
            st.text("'Base _path is not a valid directory. '")

        # Choose date
        self.use_date = st.checkbox("Choose from date?", self.use_date)
        if self.use_date:
            date = st.date_input('Choose data', self.date)
            strf_date = date.strftime(self.date_strftime)
            path = path / strf_date

        # Choose scan name
        self.use_scan_name = st.checkbox("Choose scan name", self.use_scan_name)
        if self.use_scan_name:
            if path.is_dir():
                scan_names = [x.name for x in path.iterdir() if x.is_dir()]
            else:
                st.text(
                    'No experimental run was found on that date. Consider Changing the base path '
                    'or verify the date.')
                scan_names = []
            if self.scan_name in scan_names:
                scan_name_index = scan_names.index(self.scan_name)
            else:
                scan_name_index = 0

            scan_name = st.selectbox('Choose run', scan_names, index=scan_name_index)
            if scan_name is None:
                scan_name = ""
            path = str(path / scan_name)
        self.path = path

    def streamlit_set_export_path(self):
        self.export_path = st.text_input("Save as netcdf here", value=self.export_path)
        export_path = Path(self.export_path)

        # Add date to _path
        if self.date is not None:
            self.date_to_destiny = st.checkbox("Add date to destiny _path:", self.date_to_destiny)
        else:
            self.date_to_destiny = False
        if self.date_to_destiny:
            export_path = export_path / self.strf_date / self.scan_name

        st.text("""Data will be saved here: {}""".format(str(export_path)))
        self.export_path = export_path

    def streamlit_save(self, export_path):
        if st.button('save to hdf5'):
            (export_path / 'Analysis').mkdir(parents=True, exist_ok=True)
            self.copy_sequences_variables(export_path)
            self.save_data(export_path / "raw_data")

    def streamlit_load_hdf5(self):
        load_lazy = st.checkbox("Load lazy: ", value=False)
        if st.button('load_netcdf'):
            return load_data(Path(self.export_path) / "raw_data", lazy=load_lazy)


class ImportExport:
    def __init__(self):
        self.old_structure = OldStructureStreamlit(default_path())
        self.raw_data = None

    def run(self):
        self.raw_data = self.old_structure.import_export()
        if not self.raw_data:
            st.stop()




if __name__ == '__main__':
    old_structure = OldStructureStreamlit()
    old_structure.import_export()
