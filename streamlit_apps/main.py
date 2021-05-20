r"""Streamlit app for live analysis

Usage
-------

How to start app:
run in terminal:
    streamlit run streamlit_apps\main.py --server.maxUploadSize 10000

The last number is the maximally allowed size for uploading raw_data in MBy
"""
import streamlit as st
from rydanalysis.data_structure.old_structure import OldStructure
from rydanalysis.sequence_analysis import LiveAnalysis
from streamlit_apps.import_export import page_import_export, get_default_path
from rydanalysis.auxiliary.streamlit_utils.st_state_patch import get_state


def page_set_analysis_parameters(
        old_structure=OldStructure(), raw_data=None, analyzer=LiveAnalysis()):
    if raw_data is None:
        raw_data = [None]
    analyzer.streamlit_update_params()


def main():
    state = get_state()
    if not state.old_structure:
        state.old_structure = OldStructure(get_default_path(), interface="streamlit", chunk_size=100)

    pages = {
        "Import and export data": page_import_export,
        "Analysis settings": page_set_analysis_parameters
    }

    st.sidebar.title("Rydanalysis")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


if __name__ == "__main__":
    main()
