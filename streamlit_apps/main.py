"""Streamlit app for live analysis

Usage
-------

How to start app:
run in terminal:
    streamlit run streamlit_apps\main.py --server.maxUploadSize 10000

The last number is the maximally allowed size for uploading raw_data in MBy
"""
import os
from time import sleep

os.environ['PREFECT__LOGGING__LEVEL'] = "ERROR"
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

import streamlit as st
from streamlit_apps.st_state_patch import get as get_session_state
from streamlit_apps.import_export import page_import_export
from streamlit_apps.page_single_shot import page_single_shot

import xarray as xr
import numpy as np

from streamlit_apps.image_analysis import ImageParameters
from streamlit_apps.ion_analysis import IonParameters
import rydanalysis as ra


def main():
    state = get_session_state(
        sequence_selector='by date',
        data=None,
        fit_names=['density'],
        image_parameters=ImageParameters(),
        ion_parameters=IonParameters(),
        old_structure=ra.OldStructure("", initial_update=False)
    )

    pages = {
        "Import and export data": page_import_export,
        "Single shot": page_single_shot,
    }

    st.sidebar.title("Rydanalysis")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)


if __name__ == "__main__":
    main()
