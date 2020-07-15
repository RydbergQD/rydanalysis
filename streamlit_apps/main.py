"""Streamlit app for live analysis

Usage
-------

How to start app:
run in terminal:
    streamlit run streamlit_apps\main.py --server.maxUploadSize 10000

The last number is the maximally allowed size for uploading raw_data in MBy
"""


import streamlit as st
from streamlit_apps.st_state_patch import State
import xarray as xr

from streamlit_apps.import_export import page_import_export
from streamlit_apps.single_shot import page_single_shot



def main():
    state = st.State(key="user metadata")
    if not state:
        state.sequence_selector = 'by date'
        state.path = None
        state.data = None
        state.mask = None

    pages = {
        "Import and export data": page_import_export,
        "Single shot": page_single_shot,
    }

    st.sidebar.title(":floppy_disk: Rydanalysis")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    # state.sync()


class RydState(State):
    def __init__(self):
        self.super().__init__()
        self.path = None
        self.data = None
        self.mask = None



if __name__ == "__main__":
    main()
