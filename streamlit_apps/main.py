r"""Streamlit app for live analysis

Usage
-------

How to start app:
run in terminal:
    streamlit run streamlit_apps\main.py --server.maxUploadSize 10000

The last number is the maximally allowed size for uploading raw_data in MBy
"""
import os
import time

os.environ['PREFECT__LOGGING__LEVEL'] = "ERROR"
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

import streamlit as st
from rydanalysis.auxiliary.streamlit_utils.st_state_patch import get as get_session_state
from rydanalysis.data_structure.extract_old_structure import OldStructure
from rydanalysis.sequence_analysis import LiveAnalysis
from streamlit_apps.import_export import page_import_export, get_default_path


def page_set_analysis_parameters(
        old_structure=OldStructure(), raw_data=None, analyzer=LiveAnalysis()):
    if raw_data is None:
        raw_data = [None]
    analyzer.streamlit_update_params()


def main():
    @st.cache(allow_output_mutation=True)
    def get_raw_data():
        return [None]
    raw_data = get_raw_data()

    @st.cache(allow_output_mutation=True)
    def get_old_structure():
        return OldStructure(get_default_path(), interface="streamlit")
    old_structure = get_old_structure()

    @st.cache(allow_output_mutation=True)
    def get_analyzer():
        return LiveAnalysis()
    analyzer = get_analyzer()

    pages = {
        "Import and export data": page_import_export,
        "Analysis settings": page_set_analysis_parameters
    }

    st.sidebar.title("Rydanalysis")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](old_structure, raw_data, analyzer)

    time.sleep(1)
    st.experimental_rerun()


if __name__ == "__main__":
    main()
