from distutils.dir_util import copy_tree

import streamlit as st
from pathlib import Path
import datetime
import pandas as pd
from dataclasses import dataclass
import xarray as xr
from typing import Tuple
from numbers import Number

from rydanalysis.data_structure.old_structure import OldStructure
from rydanalysis.data_structure.ryd_data import load_ryd_data


def page_import_export(state):
    st.markdown(
        r"""
        # Import and export data
        ## Import data
        
        Typical folders are:          
        data in rydberg container: \\147.142.16.121\rydberg-share\rydberg\data  
        data on axion:  
        example datasets: .\ .\samples\old_structure  
        image example: .\ .\samples\old_structure\2020_07_06\05_SNR_EIT_AT_pABSx0-8_pBlue3-6
        """)

    selector_options = pd.Series(['by path', 'by date', 'live', 'hdf5', 'example_images'])
    default_index = selector_options[selector_options == state.sequence_selector].index[0]
    sequence_selector = st.radio('Choose how to select sequence:',
                                 options=selector_options,
                                 index=int(default_index))
    state.sequence_selector = sequence_selector

    if sequence_selector == 'by date':
        state.old_structure = from_date(state.old_structure)
        data = state.old_structure.data
    elif sequence_selector == 'by path':
        state.old_structure.streamlit_from_path()
        data = state.old_structure.streamlit_update()
    elif sequence_selector == 'live':
        data = None
        st.write('Not yet implemented.')
    elif sequence_selector == 'example_images':
        # state.data = load_ryd_data(r"C:\Users\titus\Dropbox\Rydberg\Small "
        #                            r"projects\test_rydanalysis\new_structure\2020_07_06\05_SNR_EIT_AT_pABSx0-8_pBlue3"
        #                            r"-6\raw_data.h5")
        data = load_ryd_data(
            r".\.\samples\new_structure\2020_07_06\05_SNR_EIT_AT_pABSx0-8_pBlue3-6\raw_data.h5"
        )
    else:
        st.set_option('deprecation.showfileUploaderEncoding', False)
        file = st.file_uploader('choose_file')
        if not file:
            st.stop()
        data = load_ryd_data(file)

    state.data = data

    if state.data is not None:
        state.old_structure.streamlit_export()


@dataclass()
class DataImporter:
    pass


@dataclass
class OldStructureImporter(DataImporter):
    path: Path
    handle_key_errors: str = "ignore"
    sensor_widths: Tuple[Number, Number] = (1100, 214)

    def __post_init__(self):
        self.old_structure = OldStructure(self.path)

    def load(self):
        if st.button('load'):
            data = self.old_structure.data


def from_date(old_structure):
    handle_key_errors, sensor_widths = old_structure.set_init_kwargs()
    base_path = Path(st.text_input('Enter base path', value=str(old_structure.base_path)))
    if not base_path.is_dir():
        st.text("'Base path is not a valid directory. '")
        st.stop()

    # Choose date
    default_date = old_structure.date if old_structure.date else datetime.date.today()
    date: datetime.date = st.date_input('Choose data', default_date)
    strf_date = date.strftime(old_structure.date_strftime)
    date_path = base_path / strf_date
    if not date_path.is_dir():
        st.text(
            'No experimental run was found on that date. Consider Changing the base path or '
            'verify the date.')
        st.stop()
    scan_names = pd.Series([x.name for x in date_path.iterdir() if x.is_dir()])
    scan_names.insert(0, 'None')
    default_name = old_structure.scan_name if scan_names[old_structure.scan_name].index else 0
    scan_name = st.selectbox('Choose run', scan_names, index=default_name)
    if scan_name == 'None' or scan_name == old_structure.scan_name:
        st.stop()
    path = date_path / scan_name
    return OldStructure(path, handle_key_errors=handle_key_errors,
                        sensor_widths=sensor_widths)
